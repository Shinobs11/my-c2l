_deepspeed_config = {
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer":{
    "type": "Adam",
    "params": {
      "lr": 5e-5
    }
  },
  "scheduler": {
    "type": "LRRangeTest",
    "params": {
      "lr_range_test_min_lr": 1e-7,
    }
  }
}

deepspeed_config ={
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "zero_optimization": {
     "stage": 2,
     "offload_optimizer": {
         "device": "cpu",
         "pin_memory": True
     },
     "allgather_partitions": True,
     "allgather_bucket_size": 2e8,
     "reduce_scatter": True,
     "reduce_bucket_size": 2e8,
     "overlap_comm": True,
     "contiguous_gradients": True
  },
  "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
        }
    },

  "scheduler": {
      "type": "WarmupLR",
      "params": {
          "warmup_min_lr": 0,
          "warmup_max_lr": 1e-4,
          "warmup_num_steps": 500
      }
  },
}
