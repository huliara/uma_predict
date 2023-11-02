from uma_predict.bettor.fetcher import Fetcher
import numpy as np
import torch.nn.functional as F
from uma_predict.network.predictor4 import HorsePredictor
import polars as pl
from uma_predict.bettor.bettor import Bettor
np.set_printoptions(linewidth=np.inf)


train_data=pl.read_parquet("df\data10\\2016-2021-5-10.parquet")
column_len=train_data.shape[1]-18

fetcher = Fetcher(
    kaisai_nen="2023",
    kaisai_tsukihi="1029",
    race_bango="11",
    field_condition=1,
    race_name_abbre="4回東京9日",
    keibajo_code="05"
)
model = HorsePredictor.load_from_checkpoint(
    "lightning_logs/version_76_bbest/checkpoints/epoch=49-step=7150.ckpt",
    input_size=column_len,
    output_size=18,
)
model.eval()

fetcher.get_horse_data_from_jra()
print(fetcher.horse_data)
fetcher.toroku_tosu = 11
y = F.softmax(model(fetcher.horse_data[:,:])[:,:fetcher.toroku_tosu]).detach().numpy().copy()[0]
print(y)
fetcher.get_recent_odds_from_jra()
bettor = Bettor(rank_prob=y)
bettor.setup_probs()
bettor.set_EXP(fetcher)

bettor.disp_high_EXP(threshold=1.)
