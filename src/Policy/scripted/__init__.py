from .PickAndTransfer import PickAndTransferPolicy
from .Insertion import InsertionPolicy

task_name2policy = {
    "transfer_cube": PickAndTransferPolicy,
    "insertion": InsertionPolicy,
}
