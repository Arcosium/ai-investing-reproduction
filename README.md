# AI에게 투자를 그냥 맡기면 벌어 줄까?

AI에게 투자를 그냥 맡겼을 때와 과거 자료로 학습한 모델을 썼을 때의 차이를, 논문에 사용한 결과와 계산 코드로 확인한다.

상용 모델의 차트 이미지·지표 37개·열지도 조건을 LightGBM, 열지도 CNN, 가격 위치 규칙, 무작위 대조와 비교한다. 캔들 CNN은 표준화 상대수익 차로 평가한다.

## 바로 검산하기

Python 3.12에서 저장소 최상위 폴더를 기준으로 실행한다.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 code/reproduce_main_results.py
```

추가 데이터나 유료 API 호출 없이 다음을 확인한다.

- 일별 수익률에서 8개 조건의 구간별·합산 Sharpe, 누적수익률, MDD, Newey–West t값을 계산해 `performance.csv`의 모든 수치와 대조한다.
- CNN 3개 결합·MLP 두 종류의 합산 성과와 연 변동성을 별도 일별 수익률에서 검산한다. 표 8의 MLP는 `m2`이다.
- 캔들 25개 설정 × 네 구간의 100개 값을 집계한다. 음수 49개, 범위 −0.23~+0.29, 평균 −0.01, 채택 설정 0개를 확인한다.
- 점수 반전표의 원본 방향 행이 기본 성과표와 같은지 대조한다. 반대 방향 행은 저장된 평가 결과이며 이 명령으로 새로 백테스트하지 않는다.
- `dsr_inputs.json`의 기존 승자 수치는 사용하지 않고 탐색 분산 `v_pool_daily`만 가져와, 현재 CNN 일별 수익률로 DSR을 계산한다. 설정 수 142개에서 약 0.4811, 229개에서 약 0.3927이다.
- 묶음에 포함한 입력 파일의 SHA-256을 대조한다.

생성물은 `output/`에 쓴다. `results/`는 논문에 사용한 고정 결과다. MDD는 기존 평가 코드와 동일하게 누적 자산을 첫 관측값부터 계산한다. 시작 자산 1을 앞에 붙여 계산한 그림 9의 MDD와는 첫날 손실 처리에서 차이가 날 수 있다.

## 평가 구간과 지표

| 구간 | 달력 범위, 끝 날짜 미포함 | 분기 번호 |
|---|---|---:|
| A | 2024-06-01~2024-09-01 | 1 |
| B | 2024-12-01~2025-03-01 | 3 |
| C | 2025-06-01~2025-09-01 | 5 |
| D | 2025-12-01~2026-03-01 | 7 |
| 남겨 둔 기간 | 2026-03-01~2026-06-01 | 별도 평가 |

A~D를 합친 평가일은 364일, 남겨 둔 기간은 92일이다. 코호트 장부를 계산한 뒤 각 달력 구간만 잘라 합친다. 구간 사이 날짜는 수익률 표에 넣지 않는다. Sharpe는 일별 단순수익률의 평균을 표본 표준편차로 나눈 뒤 √365를 곱한다. 합산 Sharpe는 구간별 Sharpe의 평균이 아니다.

포트폴리오는 점수 상·하위 10%를 롱숏하고 각 방향에 절반씩 배분한다. 4시간봉 84개, 곧 14일을 보유하며 왕복 비용은 0.2%다. 동점의 평균 순위, 매수·매도 경계, 보유 기간이 겹치는 코호트의 자산 계산은 `code/market/paper_rules.py`와 `cohort.py`에 있다.

캔들 지표는 같은 시점 종목군 평균을 뺀 다음 보유 기간의 로그수익률을 종목별 직전 변동성으로 표준화한 뒤, 점수 상위 10% 평균에서 하위 10% 평균을 뺀 값이다. 이 값의 양수·음수는 표준화 전 투자수익률의 부호를 뜻하지 않는다. 계산은 `metrics.py`의 `evaluate`, 라벨 생성은 `train_cnn.py`에서 확인할 수 있다.

## 논문과 파일 대응

| 논문 | 고정 결과·입력 | 실행 코드와 재현 범위 |
|---|---|---|
| 캔들 절, 그림 2, 부록 설정별 표 | `results/tables/candle.json`, `candle_abcd.csv` | `reproduce_main_results.py`: 100개 값 집계, `figures.py`: 그림 2 |
| 표 6, 그림 5·8, 표 12의 주요 성과 | `results/calendar/performance.csv`, `daily_returns.csv` | `reproduce_main_results.py`: 일별 수익률부터 재계산, `figures.py`: 그림 5·8 |
| 표 8의 네 행 | 위 두 파일, `extra.json`, `data/extra_daily_returns.csv` | CNN 10개·LightGBM은 기본 표, CNN 3개·MLP는 추가 일별 자료로 검산 |
| 표 9 정보 제거 재학습 | `results/tables/ext.json` | `t21_*`의 설정별·초기화별·결합 결과. 재학습에는 원시 시세와 GPU 필요 |
| 가림 실험의 0.62·0.97 | `results/tables/occlusion_table_mean.csv`, `occlusion_table_zero.csv` | `market.saliency`: 모델 가중치·시세 필요. 두 CSV는 평균값 가림·0 가림을 구분 |
| 열지도 CNN·LightGBM 점수와 지표의 상관 | `data/score_inputs.csv.gz`, `results/tables/score_correlations.csv` | `score_corr.py`: 판단 시각별 Spearman 상관을 구간별 평균 |
| 결론의 점수 반전 결과 | `results/calendar/inverse.csv` | 원본 방향은 기본 성과와 대조. 재백테스트는 `calendar/inverse_cal.py` |
| 그림 9 | 기본 일별 수익률, `data/btc_daily_returns.csv` | `figures.py`: CNN·BTC 누적 자산 재작성 |
| 선택편향 보정 | `results/tables/dsr_inputs.json`, CNN 일별 수익률 | `reproduce_main_results.py`, `calendar/dsr_cal.py` |
| 운용 민감도·보조 분석 | `results/tables/capacity.csv`, `decay.csv`, `regime.csv`, `seeds.csv`, `stack.csv`, `light.json`, `legs.json` | `walkforward/`의 원 계산 코드. 시세·예측 원본 필요 |

`candle_settings_table.csv`는 설정의 격자·채널·변형을 기록한 기존 메타데이터 참고표다. 이 파일의 `f0~f3`는 현재 A~D 값이 아니며, 성과 인용·검산에는 **`candle.json`과 `candle_abcd.csv`만 쓴다.** `ext.json` 안의 `per`는 A·B·C·D 순서다. 가림 실험 CSV는 남겨 둔 기간의 결과이며 A~D 재학습 결과와 구분한다.

## 상관과 그림 다시 만들기

```bash
python3 code/score_corr.py
python3 -m pip install matplotlib
PAPER_FONT=/path/to/korean-font.ttf python3 code/figures.py
```

상관 계산에는 기본 의존성만 필요하다. `score_inputs.csv.gz`는 종목·판단 시각별 입력 두 개(`ret_W`, `vol_W`)와 CNN·LightGBM 점수를 묶은 자료다. `fold`는 입력 수집 때의 내부 식별자이므로 분기 번호와 다르다. 대응은 `score_corr.py`의 `SEG`에 고정했다. 수익률·변동성은 입력 창 60봉 기준이다. 상관 계산용 D 구간 자료는 2026-02-14 20:00(UTC)까지로, 판단 시각은 456개다(A·C는 각 552개, B는 540개). 논문의 상관 값도 이 자료로 계산했다.

그림은 `output/figures/`에 생긴다. `PAPER_FONT`에는 설치된 한국어 TTF·OTF 파일을 지정한다. 논문에 사용한 네 PNG는 `results/figures/`에 있으며, 같은 수치라도 서체·렌더러에 따라 픽셀은 달라질 수 있다. 폰트 파일은 포함하지 않았다.

## 시세부터 다시 실행할 때

검산용 일별 수익률과 집계 결과를 포함했다. 원시 시세, 전체 예측 NPZ, 가중치, 상용 모델 응답 원본은 포함하지 않았다. 따라서 새 학습과 모든 표의 원자료 단계 재생산에는 별도 자료가 필요하다. 상용 모델을 다시 호출하면 모델 버전과 응답의 변동으로 같은 점수가 보장되지 않는다. 자동 호출 코드는 이 묶음에 포함하지 않았다.

```bash
python3 -m pip install -r requirements-research.txt
export PYTHONPATH="$PWD/code"
export RESEARCH_DATA=/absolute/path/to/research-data
export MARKET_HISTORY="$RESEARCH_DATA/data/history"
export MARKET_BARS="$RESEARCH_DATA/work/bars_hold"
```

`RESEARCH_DATA`는 새 실행의 작업 폴더다. 원본 보관 폴더를 지정하면 출력 파일을 덮어쓸 수 있으므로 별도 작업 폴더를 쓴다. 다음 구조로 필요한 입력을 준비한다.

```text
research-data/
  data/history/base=BTC/part-2024-01.parquet   # 종목별 월간 1분봉
  work/bars_hold/{4h,1d,...}/BTC.parquet      # 해상도별 봉 캐시
  work/universe.csv                         # 학습 종목·상장폐지 상태
  work/results/                             # 설정별 예측·학습 기록
  work/results/wf8_abcd/                     # 합산·남겨 둔 기간의 코호트 장부
  work/q7fill/{mapping,jobs}.json            # 캔들 원 실험의 파일 대응
  calendar/evaluation/                       # 조건별·구간별 예측 NPZ
  calendar/evaluation_cal/                   # 새 달력 평가 결과
  results/wf8_abcd/                          # 새 집계 출력
```

1분봉에는 UTC 밀리초 `ts`, `open`, `high`, `low`, `close`, `volume`, `quote_volume` 열이 필요하다. 캐시에는 `ts,o,h,l,c,v,qv`를 쓴다. 예측 NPZ는 `ts,base,y,p,fwd,sigH` 배열로 구성된다. `p`는 중립·상승·하락 세 열이며 점수는 `p[:,1]−p[:,2]`다. `fwd`는 종목군 평균을 뺀 미래 로그수익률, `sigH`는 보유 기간에 맞춘 직전 변동성이다. 신뢰할 수 있는 NPZ만 사용한다.

실행 순서는 봉 생성 → 입력·라벨 생성과 모델 학습 → 초기화 결합 → 코호트 평가 → 달력별 절단과 검산이다. 학습 설정은 `train_cnn.py`와 `pilot_gbm.py`의 CLI로 지정한다. 실제 학습은 `RESEARCH_DATA`를 현재 작업 폴더로 삼아 실행한다. 아래 `code/...` 명령은 저장소 최상위 폴더에서 실행한다. 예를 들어 봉 생성과 개별 예측 평가는 다음과 같다.

```bash
python3 -m market.bars --bases BTC,ETH --start 2023-01 --end 2026-05 --res 4h,1d
python3 -m market.train_cnn --help
python3 -m market.pilot_gbm --help
python3 -m market.cohort --pred "$RESEARCH_DATA/calendar/evaluation/CNN_2406_08_pred.npz" --res 4h --H 84 --q 0.1 --cost 0.001
```

시세와 예측이 모두 준비되면 아래 명령으로 달력 성과표와 반전표를 만든다. 구간별 예측 파일의 이름은 `CNN`, `LLM_chart`, `LLM_context`, `LLM_heat`, `rule_zW`에 `2406_08`, `2412_02`, `2506_08`, `2512_02`, `2603_05`와 `_pred.npz`를 붙인다. 기본 모델 장부는 `evaluate_cal.py`에 명시한 `cnn10_*`, `gbm_*`, 셔플 대조 JSON 파일이 필요하다.

```bash
python3 code/calendar/evaluate_cal.py
python3 code/calendar/inverse_cal.py
python3 code/calendar/cal_extra.py
python3 code/calendar/dsr_cal.py
```

`walkforward/analyze.py`, `abcd_candle.py`, `abcd_light.py`, `legs.py`에는 원 실험의 예측 파일 배치와 보조 분석을 남겼다. 기본 평가 분기는 `1,3,5,7`이다. 이 코드의 `work/results` 파일명과 설정 매핑까지 준비해야 원시 단계 집계가 가능하다. `analyze.py`의 BTC 보조 분석에는 `$RESEARCH_DATA/work/bars_ext/4h/BTC.parquet` 캐시도 필요하다. `calendar/`의 평가 코드와 `market/`의 학습·백테스트 코드는 경로·패키지 이름을 이 묶음에 맞췄고 수치 계산식은 유지했다.

## 환경과 재현 범위

검산에 사용한 설치 버전은 `results/environment.json`에 기록했다. `requirements.txt`는 일별 자료 검산과 상관 계산용이며, `requirements-research.txt`는 학습·시세 평가용이다. GPU 학습 환경은 별도로 설치해야 한다.

상용 모델의 세 조건은 논문에서 사용한 r0 결과다. 별도 반복 실험이나 다른 달력 구간의 결과를 섞지 않았다. 포함된 일별 자료로 주요 성과표를 다시 계산할 수 있지만, 모든 실험의 재학습·유료 모델 재호출·모의 운영 장부 재실행까지 검증한 묶음은 아니다.
