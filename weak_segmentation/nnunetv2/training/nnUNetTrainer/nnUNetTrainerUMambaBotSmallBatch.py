from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch

class nnUNetTrainerUMambaBotSmallBatch(nnUNetTrainerUMambaBot):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 1200, num_of_cycles: int = 1, 
                 gamma: float = 0.8, rule: str = 'both'):
        # Override batch size
        plans['configurations'][configuration]['batch_size'] = 2
        
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, 
                        num_epochs, num_of_cycles, gamma, rule)
        
        print(f"\n{'='*80}")
        print(f"Training with batch size = 2")
        print(f"{'='*80}\n")