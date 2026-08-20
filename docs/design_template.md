# System Design Document: Multi-Agent Research System

## 1. Problem Statement

Xây dựng hệ thống trợ lý nghiên cứu tự động (**Autonomous Multi-Agent Research Assistant**) có khả năng tiếp nhận các câu hỏi nghiên cứu kỹ thuật phức tạp (ví dụ: *GraphRAG state-of-the-art*, *Kiến trúc Multi-Agent trong Customer Support*, *Production Guardrails cho LLM Agents*), tự động tìm kiếm nguồn thông tin uy tín, phân tích đánh giá đa chiều, đối chiếu quan điểm và tổng hợp thành báo cáo hoàn chỉnh có trích dẫn nguồn có thể kiểm chứng (`[i] Title (URL)`).

## 2. Why Multi-Agent?

Cách tiếp cận **Single-Agent** truyền thống (gọi một prompt LLM duy nhất) gặp các hạn chế lớn khi giải quyết bài toán nghiên cứu chuyên sâu:
1. **Context Dilution / Loãng ngữ cảnh:** Một prompt vừa tìm kiếm, vừa chắt lọc dữ liệu thô, vừa suy luận phân tích, vừa định dạng văn bản sẽ làm quá tải context window, dẫn đến suy giảm độ chi tiết và bỏ sót các khía cạnh phản biện quan trọng.
2. **Hallucination & Lack of Attribution:** Single-agent thiếu bước kiểm duyệt độc lập (`Analyst` / `Critic`), dễ sinh ra số liệu hoặc trích dẫn giả mạo (hallucinated citations).
3. **Thiếu khả năng phục hồi lỗi (Error Recovery):** Khi một thao tác truy xuất web bị lỗi, single-agent có nguy cơ hỏng toàn bộ luồng phản hồi thay vì kích hoạt cơ chế fallback như kiến trúc phân rã theo vai trò.

## 3. Agent Roles & Responsibilities

| Agent | Responsibility | Input | Output | Failure Mode & Mitigation |
|---|---|---|---|---|
| **Supervisor** | Điều phối luồng làm việc, đánh giá trạng thái hiện tại (`ResearchState`), quyết định agent tiếp theo và ngắt khi đạt điều kiện dừng | `ResearchState` (toàn bộ shared state) | `route: str` ('researcher', 'analyst', 'writer', 'critic', 'done') | *Vòng lặp vô hạn:* Bị chặn bởi `max_iterations = 6`. *Routing sai:* Fallback sang Writer khi state đủ dữ liệu. |
| **Researcher** | Tìm kiếm dữ liệu từ nguồn ngoài (Tavily/Knowledge Base), trích xuất snippet và ghi chú có cấu trúc | `request.query`, `request.max_sources` | `sources: list[SourceDocument]`, `research_notes: str` | *Không tìm thấy nguồn:* Kích hoạt domain fallback knowledge base + ghi `state.errors`. |
| **Analyst** | Đọc ghi chú nghiên cứu, bóc tách các luận điểm cốt lõi, so sánh quan điểm trái chiều, đánh giá độ tin cậy của bằng chứng | `sources`, `research_notes`, `request.query` | `analysis_notes: str` | *Thiếu nguồn để phân tích:* Trả về thông báo lỗi ngắn gọn và chuyển tiếp cho Writer tổng hợp từ context sẵn có. |
| **Writer** | Tổng hợp tri thức từ `research_notes` và `analysis_notes` thành báo cáo chuyên nghiệp theo chuẩn đối tượng độc giả, gắn kèm numbered citations | `request.audience`, `research_notes`, `analysis_notes`, `sources` | `final_answer: str` (kèm section References `[i]`) | *Quên trích dẫn:* Cơ chế tự động append danh mục `## References` nếu phát hiện thiếu marker. |
| **Critic** | (Mở rộng) Rà soát tính xác thực, kiểm tra độ phủ trích dẫn (`citation_coverage`), chấm điểm chất lượng | `final_answer`, `sources`, `analysis_notes` | `review_content`, `citation_coverage` | *Trích dẫn sai lệch:* Đánh dấu điểm trừ trong rubric chất lượng. |

## 4. Shared State Design (`ResearchState`)

| Field | Type | Mục đích & Lý do cần thiết |
|---|---|---|
| `request` | `ResearchQuery` | Chứa câu hỏi ban đầu, số lượng nguồn tối đa (`max_sources`), và đối tượng độc giả (`audience`). |
| `iteration` | `int` | Đếm số chu kỳ điều phối của Supervisor để áp dụng guardrail `max_iterations`. |
| `route_history` | `list[str]` | Nhật ký các bước handoff giữa các agent, phục vụ trace và debug trực quan. |
| `sources` | `list[SourceDocument]` | Danh sách tài liệu tham khảo chuẩn hóa (title, url, snippet, metadata). |
| `research_notes` | `str \| None` | Bản tóm tắt thô các phát hiện từ Search, là input cho Analyst. |
| `analysis_notes` | `str \| None` | Bản phân tích so sánh chuyên sâu, là input cho Writer. |
| `final_answer` | `str \| None` | Báo cáo hoàn chỉnh cuối cùng trả về cho người dùng. |
| `agent_results` | `list[AgentResult]` | Lưu trữ output và metadata (token usage, latency, cost) của từng agent riêng lẻ. |
| `trace` | `list[dict]` | Bản ghi sự kiện theo thời gian thực phục vụ observability (LangSmith/in-memory). |
| `errors` | `list[str]` | Ghi nhận lỗi phát sinh để Supervisor áp dụng chính sách xử lý sự cố. |

## 5. Routing Policy & Graph Architecture

```text
       [START]
          │
          ▼
   ┌──────────────┐
   │  Supervisor  │◄───────────────────────────┐
   └──────┬───────┘                            │
          │                                    │
          ├────► (chưa có sources) ──────────► [Researcher] ───┘
          ├────► (chưa có analysis_notes) ──► [Analyst] ──────┘
          ├────► (chưa có final_answer) ────► [Writer] ────────┘
          ├────► (đã có final_answer) ─────► [Critic] ────────┘
          └────► (iteration >= max / done) ─► [END]
```

## 6. Guardrails & Production Safety

- **Max Iterations:** Cài đặt mặc định `max_iterations = 6` (tối đa cho phép 20). Ngăn chặn triệt để vòng lặp Supervisor ↔ Worker vô hạn đốt token.
- **Execution Timeout:** Cấu hình `timeout_seconds = 60` cho từng agent API call với thư viện `tenacity` (retry tối đa 3 lần với exponential backoff).
- **Graceful Fallbacks:**
  - LLM Fallback: Tự động chuyển sang deterministic response generator nếu OpenAI API key không được cung cấp hoặc gặp sự cố mạng.
  - Search Fallback: Tự động truy vấn knowledge base offline nếu Tavily API timeout.
- **State Validation:** Toàn bộ schemas được định nghĩa bằng Pydantic V2 với `Field(ge=..., le=...)` đảm bảo type-safety và input validation chặt chẽ.

## 7. Benchmark Plan

- **Queries thử nghiệm:**
  1. *Query 1:* "Research GraphRAG state-of-the-art and write a 500-word summary" (Nghiên cứu kỹ thuật chuyên sâu).
  2. *Query 2:* "Compare single-agent and multi-agent workflows for customer support" (So sánh kiến trúc thực tế).
  3. *Query 3:* "Summarize production guardrails for LLM agents" (Thực tiễn kỹ thuật an toàn hệ thống).
- **Metrics đo lường:**
  - `Latency (s)`: Wall-clock execution time.
  - `Cost (USD)`: Ước lượng chi phí token tiêu thụ qua giá niêm yết của model.
  - `Quality Score (0-10)`: Đánh giá rubric (độ dài, cấu trúc, chiều sâu, tính xác thực).
  - `Citation Coverage (%)`: Tỷ lệ phần trăm nguồn tham khảo được trích dẫn chuẩn xác trong bài viết.
  - `Failure Rate (%)`: Tỷ lệ phần trăm truy vấn gặp lỗi nghiêm trọng hoặc thiếu kết quả.

