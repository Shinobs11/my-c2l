deepspeed_config = {
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