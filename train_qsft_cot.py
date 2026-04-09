"""FantasyVLN CoT-assisted Q-SFT Training Entry Point."""
import argparse
from swift.ray import RayHelper
from swift.llm.train.sft import SwiftSft
from swift.trainers import TrainerFactory
from swift.utils import get_logger
from data.processor_qsft_cot import load_dataset_qsft_cot

logger = get_logger()

class MyTrainerFactoryQSFT(TrainerFactory):
    TRAINER_MAPPING = {
        **TrainerFactory.TRAINER_MAPPING,
        'ummcot': 'trainer_qsft_cot.CoTQSFTTrainer',
    }

class MySwiftSftQSFT(SwiftSft):
    def _get_trainer_kwargs(self):
        kwargs = super()._get_trainer_kwargs()
        kwargs.update({
            'use_ummcot': getattr(self.args, 'use_ummcot', False),
            'q_gamma': getattr(self.args, 'q_gamma', 0.95),
            'lambda_cot': getattr(self.args, 'lambda_cot', 0.5),
            'lambda_align': getattr(self.args, 'lambda_align', 0.1),
            'label_smoothing': getattr(self.args, 'label_smoothing', 0.1),
            'clip_weight': getattr(self.args, 'clip_weight', 5.0),
            'use_ema_target': getattr(self.args, 'use_ema_target', True),
            'ema_decay': getattr(self.args, 'ema_decay', 0.995),
        })
        return kwargs
    
    def _get_dataset(self):
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        if args.dataset:
            return load_dataset_qsft_cot(args.dataset, split_dataset_ratio=args.split_dataset_ratio,
                                          shuffle=args.dataset_shuffle, **dataset_kwargs)
        return None, None

    @RayHelper.function(group='default')
    def run(self):
        train_dataset, val_dataset = self._prepare_dataset()
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        if getattr(self.args, 'use_ummcot', False):
            self.args.task_type = 'ummcot'
        trainer_cls = MyTrainerFactoryQSFT.get_trainer_cls(self.args)
        trainer = trainer_cls(model=self.model, args=self.args.training_args,
                              data_collator=self._get_data_collator(),
                              train_dataset=train_dataset, eval_dataset=val_dataset,
                              callbacks=self.callbacks, template=self.template,
                              **self._get_trainer_kwargs())
        return self.train(trainer)

def main():
    args = TrainArguments().parse_args()
    sft = MySwiftSftQSFT(args)
    result = sft.run()
    logger.info(f'Training complete: {result}')

if __name__ == '__main__':
    main()
