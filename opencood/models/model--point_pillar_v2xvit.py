PointPillarTransformer(
  (pillar_vfe): PillarVFE(
    (pfn_layers): ModuleList(
      (0): PFNLayer(
        (linear): Linear(in_features=10, out_features=64, bias=False)
        (norm): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      )
    )
  )
  (scatter): PointPillarScatter()
  (backbone): BaseBEVBackbone(
    (blocks): ModuleList(
      (0): Sequential(
        (0): ZeroPad2d((1, 1, 1, 1))
        (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (2): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (8): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (9): ReLU()
        (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (12): ReLU()
      )
      (1): Sequential(
        (0): ZeroPad2d((1, 1, 1, 1))
        (1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (9): ReLU()
        (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (12): ReLU()
        (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (15): ReLU()
        (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (18): ReLU()
      )
      (2): Sequential(
        (0): ZeroPad2d((1, 1, 1, 1))
        (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (2): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (6): ReLU()
        (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (8): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (9): ReLU()
        (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (11): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (12): ReLU()
        (13): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (14): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (15): ReLU()
        (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (17): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (18): ReLU()
        (19): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (20): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (21): ReLU()
        (22): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (23): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (24): ReLU()
        (25): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (26): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (27): ReLU()
      )
    )
    (deblocks): ModuleList(
      (0): Sequential(
        (0): ConvTranspose2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): Sequential(
        (0): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): Sequential(
        (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(4, 4), bias=False)
        (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
  )
  (shrink_conv): DownsampleConv(
    (layers): ModuleList(
      (0): DoubleConv(
        (double_conv): Sequential(
          (0): Conv2d(384, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (1): ReLU(inplace=True)
          (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU(inplace=True)
        )
      )
    )
  )
  (fusion_net): V2XTransformer(
    (encoder): V2XTEncoder(
      (sttf): STTF()
      (prior_feed): Linear(in_features=259, out_features=256, bias=True)
      (layers): ModuleList(
        (0): ModuleList(
          (0): V2XFusionBlock(
            (layers): ModuleList(
              (0): ModuleList(
                (0): PreNorm(
                  (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (fn): HGTCavAttention(
                    (attend): Softmax(dim=-1)
                    (drop_out): Dropout(p=0.3, inplace=False)
                    (k_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (q_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (v_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (a_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (norms): ModuleList()
                  )
                )
                (1): PreNorm(
                  (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (fn): PyramidWindowAttention(
                    (pwmsa): ModuleList(
                      (0): BaseWindowAttention(
                        (to_qkv): Linear(in_features=256, out_features=768, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=256, out_features=256, bias=True)
                          (1): Dropout(p=0.3, inplace=False)
                        )
                      )
                      (1): BaseWindowAttention(
                        (to_qkv): Linear(in_features=256, out_features=768, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=256, out_features=256, bias=True)
                          (1): Dropout(p=0.3, inplace=False)
                        )
                      )
                      (2): BaseWindowAttention(
                        (to_qkv): Linear(in_features=256, out_features=768, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=256, out_features=256, bias=True)
                          (1): Dropout(p=0.3, inplace=False)
                        )
                      )
                    )
                    (split_attn): SplitAttn(
                      (fc1): Linear(in_features=256, out_features=256, bias=False)
                      (bn1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                      (act1): ReLU()
                      (fc2): Linear(in_features=256, out_features=768, bias=False)
                      (rsoftmax): RadixSoftmax()
                    )
                  )
                )
              )
            )
          )
          (1): PreNorm(
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (fn): FeedForward(
              (net): Sequential(
                (0): Linear(in_features=256, out_features=256, bias=True)
                (1): GELU(approximate='none')
                (2): Dropout(p=0.3, inplace=False)
                (3): Linear(in_features=256, out_features=256, bias=True)
                (4): Dropout(p=0.3, inplace=False)
              )
            )
          )
        )
        (1): ModuleList(
          (0): V2XFusionBlock(
            (layers): ModuleList(
              (0): ModuleList(
                (0): PreNorm(
                  (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (fn): HGTCavAttention(
                    (attend): Softmax(dim=-1)
                    (drop_out): Dropout(p=0.3, inplace=False)
                    (k_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (q_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (v_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (a_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (norms): ModuleList()
                  )
                )
                (1): PreNorm(
                  (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (fn): PyramidWindowAttention(
                    (pwmsa): ModuleList(
                      (0): BaseWindowAttention(
                        (to_qkv): Linear(in_features=256, out_features=768, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=256, out_features=256, bias=True)
                          (1): Dropout(p=0.3, inplace=False)
                        )
                      )
                      (1): BaseWindowAttention(
                        (to_qkv): Linear(in_features=256, out_features=768, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=256, out_features=256, bias=True)
                          (1): Dropout(p=0.3, inplace=False)
                        )
                      )
                      (2): BaseWindowAttention(
                        (to_qkv): Linear(in_features=256, out_features=768, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=256, out_features=256, bias=True)
                          (1): Dropout(p=0.3, inplace=False)
                        )
                      )
                    )
                    (split_attn): SplitAttn(
                      (fc1): Linear(in_features=256, out_features=256, bias=False)
                      (bn1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                      (act1): ReLU()
                      (fc2): Linear(in_features=256, out_features=768, bias=False)
                      (rsoftmax): RadixSoftmax()
                    )
                  )
                )
              )
            )
          )
          (1): PreNorm(
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (fn): FeedForward(
              (net): Sequential(
                (0): Linear(in_features=256, out_features=256, bias=True)
                (1): GELU(approximate='none')
                (2): Dropout(p=0.3, inplace=False)
                (3): Linear(in_features=256, out_features=256, bias=True)
                (4): Dropout(p=0.3, inplace=False)
              )
            )
          )
        )
        (2): ModuleList(
          (0): V2XFusionBlock(
            (layers): ModuleList(
              (0): ModuleList(
                (0): PreNorm(
                  (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (fn): HGTCavAttention(
                    (attend): Softmax(dim=-1)
                    (drop_out): Dropout(p=0.3, inplace=False)
                    (k_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (q_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (v_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (a_linears): ModuleList(
                      (0): Linear(in_features=256, out_features=256, bias=True)
                      (1): Linear(in_features=256, out_features=256, bias=True)
                    )
                    (norms): ModuleList()
                  )
                )
                (1): PreNorm(
                  (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (fn): PyramidWindowAttention(
                    (pwmsa): ModuleList(
                      (0): BaseWindowAttention(
                        (to_qkv): Linear(in_features=256, out_features=768, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=256, out_features=256, bias=True)
                          (1): Dropout(p=0.3, inplace=False)
                        )
                      )
                      (1): BaseWindowAttention(
                        (to_qkv): Linear(in_features=256, out_features=768, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=256, out_features=256, bias=True)
                          (1): Dropout(p=0.3, inplace=False)
                        )
                      )
                      (2): BaseWindowAttention(
                        (to_qkv): Linear(in_features=256, out_features=768, bias=False)
                        (to_out): Sequential(
                          (0): Linear(in_features=256, out_features=256, bias=True)
                          (1): Dropout(p=0.3, inplace=False)
                        )
                      )
                    )
                    (split_attn): SplitAttn(
                      (fc1): Linear(in_features=256, out_features=256, bias=False)
                      (bn1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                      (act1): ReLU()
                      (fc2): Linear(in_features=256, out_features=768, bias=False)
                      (rsoftmax): RadixSoftmax()
                    )
                  )
                )
              )
            )
          )
          (1): PreNorm(
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (fn): FeedForward(
              (net): Sequential(
                (0): Linear(in_features=256, out_features=256, bias=True)
                (1): GELU(approximate='none')
                (2): Dropout(p=0.3, inplace=False)
                (3): Linear(in_features=256, out_features=256, bias=True)
                (4): Dropout(p=0.3, inplace=False)
              )
            )
          )
        )
      )
      (rte): RTE(
        (emb): RelTemporalEncoding(
          (emb): Embedding(100, 256)
          (lin): Linear(in_features=256, out_features=256, bias=True)
        )
      )
    )
  )
  (cls_head): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
  (reg_head): Conv2d(256, 14, kernel_size=(1, 1), stride=(1, 1))
)