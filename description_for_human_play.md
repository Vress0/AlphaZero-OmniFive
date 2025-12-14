## human_play 說明

例如你想要與 "best_policy_6_6_4.model1" 這個模型遊玩
就先根據這個模型所適用的棋盤與 resblock, channel 進行適配
首先打開 config.json， 看到後面的 human_play 欄位，裡面的"model_file"，把它改成 "best_policy_6_6_4_64x3/best_policy_6_6_4.model1" 

```json
"human_play": {
    "model_file": "best_policy_6_6_4_64x3/best_policy_6_6_4.model1"
    (paras)...
}
```
這樣就可以讀取到這個資料夾的 best_policy_6_6_4.model1 模型，然後因為這個模型只能在 6*6 四子棋下的規則運行
所以再繼續看到 config.json 的 board 欄位，填入長寬與 n_in_row

```json
  "board": {
    "width": 9,
    "height": 9,
    "n_in_row": 5
}
```
最後還要調整 residual block, channel 數量才能正常運行，像是這個現成模型的資料夾名字
best_policy_6_6_4_64x3，代表他的 channel 與 residual block 數量是 64 與 3

```json
  "network": {
    "num_channels": 64,
    "num_res_blocks": 3
}
```

最後在 terminal 執行 python human_play.py 就能運行了