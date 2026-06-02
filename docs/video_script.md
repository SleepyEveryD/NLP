# 5-min video script — Who Wants to Be a PoliMillionaire?

**Constraints (PDF):** ≤5 minutes · screen-capture of the *notebook* (no slides) · both members speak ·
**must name the RAG APIs out loud.** Presenters: **J = Jiaxin Yang**, **R = Runjie (Simone) Dai**.

> 用法:左边是时间轴 + 屏幕动作(中文提示),右边「»」是照念的英文旁白。语速放慢一点,5 分钟≈700 词,留 buffer。
> 录之前先把 6 个 notebook 跑一遍存好输出(尤其 03 的 leaderboard sweep、05 的 10 轮 + 消融图),这样录屏时直接滚动展示已有结果,不用现场等。

---

## 0:00 – 0:25 · Intro  — **J**
*屏幕:notebook 01 顶部的 header(组员/视频/声明)*

» Hi, we're Jiaxin and Runjie. Our project is a chatbot that plays the quiz *Who Wants to Be a
PoliMillionaire?*. The rules: the model must run **locally** — an open-weight model, no external LLM APIs —
and it has **30 seconds** per question. We use **Qwen2.5-7B-Instruct in 4-bit** on a Colab T4, greedy decoding.
We talk to the game through the **text API** from Tutorial 7.

## 0:25 – 1:05 · Architecture — **J**
*屏幕:notebook 01,滚到「🔬 核心实现:QAPipeline.answer() 七段式编排」*

» Everything goes through one orchestrator, `QAPipeline`. A question runs **seven stages**: classify the
question, an optional deterministic **solver**, optional **retrieval**, build the **prompt**, **generate**,
an optional **tool**, and **parse** the letter out. Each stage is timed against the 30-second wall and the
whole thing is crash-safe — in live play we must always submit *something*.
*滚到 03 的 `RACE_PIPELINES` cell*
» Key design: **per-race pipelines** — every competition gets its **own** pipeline, so tuning one race
never leaks into another.

## 1:05 – 1:45 · Prompt engineering — **R**
*屏幕:notebook 02,滚到「🔬 核心实现:prompt 策略库 + cot_v2」与对比图*

» We benchmarked three prompt families — zero-shot, few-shot, and chain-of-thought. Our workhorse is
**`cot_v2`**, and it was driven by two real failures we logged. Once the model reasoned correctly but picked
the wrong option because two options shared a conclusion — so `cot_v2` forces it to match **every number**,
not just the conclusion. Another time it wrote pages of LaTeX and hit the token cap **before** the answer
line — so `cot_v2` **bans LaTeX, caps the steps, and forces an `Answer:` line**. The comparison here shows
Maths is where prompts matter most.

## 1:45 – 2:35 · Maths — solver + adaptive routing — **J**
*屏幕:notebook 04,滚到「🔬 核心实现:自适应路由 + 确定性解法器」,再到热力图/实验表*

» Maths is the hardest race for a 7B model, so it gets special treatment. First, a **deterministic solver**
runs *before* the LLM — for types we can compute exactly: finite-field roots, gcd, ring characteristic,
percentages, weekdays and more. If it's confident, we answer from it and skip the model entirely.
» Otherwise we use **adaptive prompt routing**: a classifier tags the reasoning shape, and only genuine
**time-interval counting** questions get the structured-enumeration prompt — everything else stays on
`cot_v2`, which never truncates at the wall. This notebook 04 study — four conditions over forty questions —
is what justified that conservative policy. Maths now reaches **level 11**.

## 2:35 – 3:10 · Agentic AI — voting + calculator — **R**
*屏幕:notebook 03,滚到「🔬 核心实现:多数投票 + 安全计算器」*

» We also use agentic techniques. **Majority voting** combines several predictions into one — either
several sampled chains of one model, or different models — and the confidence becomes the **vote share**, a
real calibration signal. And our **calculator** is a tool the model can call, but it's a **safe AST
evaluator** — we never give the model `eval()`. It acts as a *verifier*: it only overrides the answer when
the computed number **uniquely matches one option**.

## 3:10 – 4:05 · News — live RAG (NAME THE APIs) — **R**
*屏幕:notebook 05,滚到「🔬 核心实现:News 实时 RAG 检索级联」,再到 10 轮结果 + RAG vs no-RAG 消融*

» News asks about events **after the model's training cutoff**, so it needs real-time retrieval — using only
**free, raw, non-generated content**, as the rules require. Let me name the sources we use:
**Google News RSS** for headlines, the **Guardian Content API** for raw article body text, and a
**headless Chromium browser via Playwright** to open other articles past the consent wall. We also fall back
to **Wikipedia** and a local **FAISS** index over Simple-Wikipedia.
» The retriever is a **cascade**: a date-windowed headline query, then keyword and option re-searches if it
comes up off-topic, then a **recency re-rank** by publication date, then bodies — Guardian first, browser
otherwise — with a per-document relevance filter. This **RAG-vs-no-RAG** ablation shows retrieval genuinely
helps: the model's own prior is often wrong. News now reaches **level 12**.

## 4:05 – 4:35 · Results & evaluation — **J**
*屏幕:notebook 03 的 sweep 结果 / leaderboard 截图*

» Across the leaderboard: **Entertainment, History, Science, and Philosophy are all maxed at level 15**;
**Maths is at level 11** and **News reaches level 12** at around 82–85% per question. Beyond the
leaderboard we evaluated per-topic accuracy, latency against the 30-second wall, prompt sensitivity, the
RAG ablation, and a math model comparison — all in the notebooks.

## 4:35 – 5:00 · Wrap — **J + R**
*屏幕:回到 header*

» **(R)** To be transparent: we used a coding assistant to help write and refactor parts of the code, but
the design, the experiments, and the analysis are ours — the assignment was not handed to an LLM.
» **(J)** That's our PoliMillionaire agent — local models, per-race strategies, agentic tools, and live RAG.
Thanks for watching!

---

### 录制小贴士
- 英文旁白(课程/面试是英文)。若你们想用中文讲也可以,逐段翻译即可。
- 每人对半开口,符合「members of the group present」。
- 严格卡 5:00;超时就砍 4:05–4:35 的细节,但 **News 点名 API 那段(3:10–4:05)绝不能删**——那是硬性规则。
- 录屏全程展示 *已经跑好输出* 的 notebook,边滚边讲,不要现场等模型。
