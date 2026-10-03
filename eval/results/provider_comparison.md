### 99 questions

|                                         |   faithfulness |   answer_rel |   ctx_precision |   ctx_recall |   refusal_acc |   data_fact |   cited |   multi_src |   abstain |   grader_pass |   $/question |   tokens/q |   p50_s |   p95_s |   false_refusals |
|:----------------------------------------|---------------:|-------------:|----------------:|-------------:|--------------:|------------:|--------:|------------:|----------:|--------------:|-------------:|-----------:|--------:|--------:|-----------------:|
| deepseek-flash (deepseek/deepseek-chat) |          0.946 |        0.863 |           0.826 |         0.92 |             1 |           1 |   0.983 |       0.667 |         1 |          0.99 |        0.001 |    4686.25 |    4.26 |   12.41 |                0 |

### 69 questions

|                                                            |   faithfulness |   answer_rel |   ctx_precision |   ctx_recall |   refusal_acc |   data_fact |   cited |   multi_src |   abstain |   grader_pass |   $/question |   tokens/q |   p50_s |   p95_s |   false_refusals |
|:-----------------------------------------------------------|---------------:|-------------:|----------------:|-------------:|--------------:|------------:|--------:|------------:|----------:|--------------:|-------------:|-----------:|--------:|--------:|-----------------:|
| claude-haiku-4-5-n69 (anthropic/claude-haiku-4-5-20251001) |          0.934 |        0.89  |           0.792 |        0.842 |         0.984 |         1   |   0.967 |           1 |         1 |         1     |        0.004 |    4220.29 |    3.92 |    7.91 |                1 |
| claude-sonnet-5-5-n69 (anthropic/claude-sonnet-5-5)        |          0.916 |        0.797 |           0.725 |        0.9   |         1     |         1   |   1     |           1 |         1 |         1     |        0.011 |    4930.04 |    6.9  |   10.63 |                0 |
| deepseek-flash-n69 (deepseek/deepseek-chat)                |          0.908 |        0.82  |           0.754 |        0.883 |         1     |         1   |   1     |           1 |         1 |         1     |        0.001 |    3784.81 |    4.23 |    8.88 |                0 |
| gpt-4o-mini-n69 (openai/gpt-4o-mini)                       |          0.826 |        0.876 |           0.784 |        0.969 |         0.935 |         0.9 |   0.867 |           1 |         1 |         0.986 |        0     |    3422.48 |    4.61 |   16.33 |                4 |

### 20 questions

|                                            |   faithfulness |   answer_rel |   ctx_precision |   ctx_recall |   refusal_acc |   data_fact |   cited |   multi_src |   abstain |   grader_pass |   $/question |   tokens/q |   p50_s |   p95_s |   false_refusals |
|:-------------------------------------------|---------------:|-------------:|----------------:|-------------:|--------------:|------------:|--------:|------------:|----------:|--------------:|-------------:|-----------:|--------:|--------:|-----------------:|
| multi-company-fix (deepseek/deepseek-chat) |            nan |          nan |             nan |          nan |             1 |         nan |       1 |           1 |       nan |             1 |        0.002 |       5297 |    5.49 |   21.82 |                0 |
