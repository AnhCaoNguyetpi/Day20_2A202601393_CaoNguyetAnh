# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

### 1. Case nào nên dùng multi-agent? Vì sao?
- **Nên dùng:**
  - **Nghiên cứu tài liệu đa tầng (Deep Technical Research):** Cần phân tách rõ ràng việc thu thập dữ liệu thô (Search), đánh giá logic/so sánh chéo (Analyst), và biên tập định dạng văn phong kèm trích dẫn (Writer).
  - **Hệ thống có nhiều công cụ & quyền hạn riêng biệt (Role-based Tool Access):** Ví dụ trong Customer Support, Agent Triage chỉ phân loại, Agent Billing có quyền truy cập DB thanh toán, Agent Technical truy cập docs kỹ thuật.
  - **Quy trình đòi hỏi kiểm duyệt độc lập (Independent Verification / Human-in-the-loop):** Có bước Critic hoặc Reviewer chấm điểm và bắt buộc Agent Writer sửa lại nếu chưa đạt rubric hoặc citation coverage dưới ngưỡng.
- **Vì sao:** Tránh loãng ngữ cảnh (context saturation), giảm thiểu ảo giác (hallucination), dễ cô lập lỗi (isolation) và cho phép tối ưu prompt/model riêng cho từng vai trò (chọn model nhỏ rẻ cho search/triage, model lớn cho reasoning).

### 2. Case nào không nên dùng multi-agent? Vì sao?
- **Không nên dùng:**
  - **Truy vấn đơn giản, độ trễ thấp (Sub-second Latency Queries):** Tóm tắt một đoạn văn bản ngắn, dịch thuật, định dạng JSON đơn giản, FAQ trực tiếp.
  - **Quy trình tuần tự tuyến tính không có điều kiện rẽ nhánh (Simple Linear Pipelines):** Khi bài toán chỉ là Prompt A -> Prompt B cố định, dùng single-agent hoặc chain tuần tự đơn giản sẽ hiệu quả hơn nhiều so với việc duy trì đồ thị trạng thái phức tạp.
  - **Dự án ngân sách token/chi phí hạn hẹp:** Multi-agent tiêu thụ gấp 3-5 lần token do chi phí truyền tải context giữa các node trong shared state.
- **Vì sao:** Gây lãng phí chi phí token không cần thiết, tăng độ trễ (latency overhead) do nhiều roundtrips LLM, và tăng độ phức tạp trong việc debug đồ thị trạng thái.

