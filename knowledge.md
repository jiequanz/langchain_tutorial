Jun 13, 2026

AI Agent / LangChain & LangGraph
Interview Glossary  面试术语表
Each term: English definition + everyday analogy, then a Chinese explanation and example.  每个术语：英文定义 + 生活比喻，随后是中文解释和例子。
Foundational LLM Concepts  基础大模型概念
Token / tokenization（词元 / 分词） — the sub-word pieces a model reads and is billed by.
Think of it like: The AI can't read a whole word in one gulp, so it chops text into little pieces called tokens, like breaking a chocolate bar into squares before eating. "Unbelievable" becomes "un" + "believ" + "able". Just like a shop charges per square of chocolate, the AI charges per token.
中文解释：模型读取和计费的最小文字单位。
打个比方：AI 不能一口吞下整个单词，所以把文字切成小块，叫"词元"，就像把一整块巧克力掰成小方块再吃。"unbelievable"会被切成"un"+"believ"+"able"。就像商店按巧克力方块数收钱，AI 也按词元数收费。
Context window（上下文窗口） — the most tokens the model can look at in one go.
Think of it like: Imagine a desk that only holds so many papers. Pile on too many and the bottom ones fall off the edge. The AI's "desk" is its context window: once full, the oldest notes slide off and are forgotten.
中文解释：模型一次能看的最多词元数量。
打个比方：想象一张桌子只能放下有限的纸，放太多最下面的就会掉下去。AI 也有这样一张"桌子"，叫上下文窗口；一旦放满，最早的笔记就会滑落、被忘记。
Inference（推理） — running a finished model to get an answer (not training it).
Think of it like: Training is like a student studying all year. Inference is test day, when the student actually answers using what they learned. Every time you ask the AI and it replies, that's inference, not studying.
中文解释：用训练好的模型生成答案（而不是训练它）。
打个比方：训练像学生用一整年学习；推理是考试当天，用学到的东西真正答题。你每次问 AI、它回答的那一刻，就是推理，而不是学习。
Temperature / top-p（温度 / top-p） — dials that control how random the output is.
Think of it like: Think of an ice-cream machine with a "surprise" dial. Turn it to 0 and you get plain vanilla every time. Turn it up high and it picks wild random flavors. Low = safe and predictable; high = creative and surprising.
中文解释：控制输出随机程度的旋钮。
打个比方：想象一台带"惊喜"旋钮的冰淇淋机。转到 0，每次都给你原味香草；调高，它就挑各种古怪口味。温度低=稳妥可预测；温度高=有创意、爱给惊喜。
Message roles（消息角色） — the system / user / assistant parts of a chat.
Think of it like: Like a school play where everyone has a part. The "system" is the director who says "you're a friendly teacher." The "user" is you asking. The "assistant" is the AI saying its lines back.
中文解释：对话里的 system / user / assistant 三种身份。
打个比方：像一场校园话剧，每人有角色。"system"是导演，说"你是一位友善的老师"；"user"是你提问；"assistant"是 AI 说出它的台词。
Hallucination（幻觉） — a confident answer that is actually made up.
Think of it like: Picture a kid who doesn't know a test answer but confidently writes something that sounds right. The AI does this too: it can invent a "fact," like naming a book that was never written, and say it as if it's true.
中文解释：听起来很自信、其实是编造的答案。
打个比方：想象一个孩子不知道考题答案，却很自信地写下听上去对的东西。AI 也会这样：它可能"编"出一个事实，比如说出一本根本不存在的书，还当成真的说出来。
In-context learning (zero/few-shot)（上下文学习 / 少样本学习） — teaching by examples in the prompt, no retraining.
Think of it like: Like showing a kid how to sort two socks yourself, then they copy you for the rest, no "sock school" needed. You put a few examples right in your question and the AI copies the pattern.
中文解释：在提示里给例子来教它，不用重新训练。
打个比方：就像你先帮孩子配好两双袜子做示范，他就照着把剩下的配完，不用去上"袜子学校"。你在问题里放几个例子，AI 就照着模仿。
Chain-of-thought (CoT)（思维链） — asking the model to reason step by step.
Think of it like: When a math problem is tricky, the teacher says "show your work." Thinking out loud, one step at a time, means fewer mistakes. Telling the AI to "think step by step" makes it work through the problem.
中文解释：让模型一步步推理。
打个比方：数学题难的时候，老师会说"把过程写出来"。一步步想，错得少。让 AI"一步步想"，它就会推导，而不是脱口而出答案。
Structured output（结构化输出） — forcing a tidy JSON/form-shaped answer.
Think of it like: Instead of telling a long story about your toy, you fill in a neat form: Name: teddy, Color: brown, Size: big. Structured output makes the AI fill in a tidy form so a computer can read it easily.
中文解释：强制模型输出整齐的 JSON/表格式答案。
打个比方：与其讲一长段关于你玩具的故事，不如填一张整齐的表：名字：泰迪；颜色：棕色；大小：大。结构化输出就是让 AI 填整齐的表，方便电脑读取。
RAG vs fine-tuning vs long-context（RAG / 微调 / 长上下文） — three ways to give the AI new knowledge.
Think of it like: RAG is an open-book test: the AI peeks at the right library book while answering. Fine-tuning is sending it to school to truly learn a skill. Long-context is handing it the one page it needs right before you ask.
中文解释：给 AI 补充新知识的三种方法。
打个比方：RAG 是开卷考试：答题时偷看正确的那本书。微调是送它去上学、真正学会一项新技能。长上下文是直接把它需要的那一页递给它再提问。
RAG (Retrieval-Augmented Generation)  检索增强生成
Indexing vs query phase（建索引阶段 / 查询阶段） — build the searchable store once; then answer fast each time.
Think of it like: A librarian spends the weekend shelving every book with labels (indexing). Then when you ask, she instantly grabs the right book (querying). The hard sorting happens once; answering is then quick.
中文解释：先一次性建好可搜索的库，之后每次都能快速回答。
打个比方：图书管理员花一个周末把每本书摆到正确书架并贴标签（建索引）。之后你一提问，她立刻走到对应书架拿书（查询）。难的整理只做一次，回答就很快。
Chunking (size / overlap)（分块 / 块大小 / 重叠） — cutting documents into bite-size pieces to embed.
Think of it like: A whole pizza is too big for one bite, so you cut it into slices. The AI slices documents into "chunks." Overlap means each slice shares a little crust with the next, so nothing important is lost at the cut.
中文解释：把文档切成便于嵌入的小块。
打个比方：一整张披萨一口吃不下，所以切成片。AI 把大文档切成"块"。重叠是指每片和下一片共享一点饼边，免得切口处的重要内容丢失。
Chunking strategies（分块策略） — different ways to split (fixed, recursive, semantic, parent-document...).
Think of it like: Different ways to cut the pizza: equal squares, along the natural lines, or take a tiny bite but keep the whole slice nearby. Each cutting style trades off how well you find and understand the pieces.
中文解释：不同的切分方式（固定、递归、语义、父文档……）。
打个比方：切披萨的不同方法：切成相等方块、沿天然纹路切、或者咬一小口但把整片留在旁边以备需要。每种切法在"好不好找、好不好懂"上各有取舍。
Contextual retrieval（上下文检索） — add a context label to each chunk before embedding.
Think of it like: A sticky note that just says "30 days" is useless alone. So you add a label: "Return policy: 30 days." Now anyone who finds it understands. The AI tags each chunk so it makes sense even when found alone.
中文解释：嵌入前给每个块加上下文标签。
打个比方：一张只写着"30 天"的便利贴单独看毫无意义。于是你加个标签："退货政策：30 天"。这样谁找到都懂。AI 给每个块贴上小标签，即使单独被找到也讲得通。
Top-k（Top-k / 取前 k 个） — how many chunks the retriever brings back.
Think of it like: When you ask the librarian, do you want the 1 best book or the top 3? "Top-k" is that number. k=3 means "bring me the 3 closest matches to my question."
中文解释：检索器一次返回多少个块。
打个比方：你找管理员帮忙，是要 1 本最好的书，还是前 3 本最好的？"Top-k"就是这个数字。k=3 意思是"给我和问题最接近的 3 个"。
MMR (maximal marginal relevance)（最大边际相关性） — pick relevant but varied chunks, not duplicates.
Think of it like: If you ask for dog books, you don't want 3 copies of the same one; you want 3 different helpful ones. MMR makes the AI grab varied chunks instead of near-identical twins.
中文解释：既相关又多样，避免重复。
打个比方：你要狗的书，不想拿到 3 本一模一样的，而想要 3 本不同的好书。MMR 让 AI 取到多样的块，而不是几乎一样的"双胞胎"。
Hybrid search（混合检索） — mix meaning-based and exact-keyword search.
Think of it like: Two ways to find a book: by what it's about, or by its exact title words. Hybrid uses both, so you don't miss a book that uses a special exact name like a serial number "X-200."
中文解释：结合"按意思"和"按关键词"两种搜索。
打个比方：找书有两种方式：按内容意思，或按确切的书名词。混合检索两者都用，这样就不会漏掉用了特殊确切名字（比如序列号"X-200"）的书。
Reranking（重排序） — a careful second pass that keeps only the best chunks.
Think of it like: First you grab a big armful of maybe-useful books (say 50). Then a picky helper looks closely and keeps only the 5 best. Reranking is that careful second sort that tosses the so-so ones.
中文解释：再仔细排一遍，只留最好的块。
打个比方：先抱回一大堆可能有用的书（比如 50 本），再让一个挑剔的助手仔细看，只留最好的 5 本给你。重排序就是那次仔细的二次筛选，扔掉一般般的。
Query rewriting / multi-query / HyDE（查询改写 / 多查询 / HyDE） — reshape a vague question for better searching.
Think of it like: A kid asks "what about the blue one?" and you have no idea what they mean. So you turn it into a full question: "How much does the blue bike cost?" The AI rewrites fuzzy questions into clear ones before searching.
中文解释：把含糊的问题改写得更好搜。
打个比方：小孩问"那蓝色的那个呢？"你根本不知道指什么。于是你把它变成完整问题："那辆蓝色自行车多少钱？"AI 在搜索前先把模糊问题改写成清楚完整的。
Grounding / faithfulness（有据可依 / 忠实度） — the answer must come from the retrieved text.
Think of it like: Like a kid only allowed to answer using the book in front of them, no made-up extras. "Grounded" means the answer comes straight from the real pages. If it's not in the book, the honest answer is "I don't know."
中文解释：答案必须来自检索到的文字。
打个比方：像一个只能用眼前这本书回答、不能瞎编的孩子。"有据"就是答案直接来自真实的书页。如果书里没有，诚实的回答就是"我不知道"。
Corrective / Self / Agentic RAG（纠正式 / 自我 / 智能体 RAG） — grade what you found and retry if it's bad.
Think of it like: You look for your toy, check a box, it's not there, so you stop, think "where else?", and check a better box. The AI does this: if the pages found don't help, it changes its search and tries again instead of guessing.
中文解释：给检索结果打分，不好就重试。
打个比方：你找玩具，翻开一个箱子没有，于是停下想"还能在哪？"，再翻一个更可能的箱子。AI 也这样：找到的页帮不上忙，就改搜索再试，而不是瞎猜。
Embeddings & Vector Search  嵌入与向量检索
Embedding（嵌入 / 向量） — a list of numbers that captures a text's meaning.
Think of it like: Imagine every word gets a house on a giant map, and similar-meaning words live close together. "Dog" and "puppy" are next-door neighbors; "dog" and "tax" live in different cities. An embedding is a word's address on that meaning-map, in numbers.
中文解释：一串能表示文字含义的数字。
打个比方：想象每个词在一张大地图上都有一栋房子，意思相近的词住得近。"狗"和"小狗"是邻居；"狗"和"税"住在不同城市。嵌入就是一个词在这张"含义地图"上的地址，用数字写出来。
Embedding model（嵌入模型） — the tool that turns text into those vectors.
Think of it like: It's the mapmaker, the helper that decides where each word or sentence goes on the meaning-map. It's a different helper from the one that writes the answers.
中文解释：把文字变成向量的工具。
打个比方：它是"绘图员"，决定每个词或句子在含义地图上放哪。它和负责写答案的那个助手是不同的工具。
Vector store / database（向量库 / 向量数据库） — where embeddings are saved and searched.
Think of it like: A giant address book that remembers where everything lives on the meaning-map, so you can quickly look up the nearest neighbors. Chroma, Pinecone, pgvector are like different brands of it.
中文解释：保存并搜索嵌入向量的地方。
打个比方：一本巨大的通讯录，记住含义地图上每样东西住在哪，方便你快速查到最近的邻居。Chroma、Pinecone、pgvector 就像这本通讯录的不同品牌。
Cosine similarity（余弦相似度） — a score for how close two vectors are.
Think of it like: A way to measure how close two houses are on the meaning-map. A high score like 0.95 means "right next door, almost the same"; a low score means "far away, totally different."
中文解释：衡量两个向量有多接近的分数。
打个比方：一种衡量含义地图上两栋房子有多近的方法。0.95 这样的高分表示"就在隔壁，几乎一样"；低分表示"离得很远，完全不同"。
ANN / HNSW（近似最近邻 / HNSW） — fast approximate nearest-neighbor search.
Think of it like: With a million houses, checking every one to find the closest takes forever. ANN is a clever shortcut that finds very-close neighbors super fast, like using highways instead of walking every street.
中文解释：快速的近似最近邻搜索。
打个比方：要是有一百万栋房子，挨个检查找最近的会花很久。ANN 是个聪明的捷径，能极快找到很近的邻居，就像走高速而不是把每条街都走一遍。
Dimensionality（维度） — how many numbers are in each vector.
Think of it like: A simple map uses 2 directions (left-right, up-down). The meaning-map uses hundreds at once to capture detail. More directions means it can tell apart more tiny differences in meaning.
中文解释：每个向量里有多少个数字。
打个比方：简单地图只用 2 个方向（左右、上下）。含义地图同时用上百个方向来表达更多细节。维度越多，越能分辨出意思上的细微差别。
Agents & Tool Use  智能体与工具使用
Agent（智能体） — an LLM that can take actions in a loop, not just chat.
Think of it like: A plain AI only talks. An agent can also do things, like a helper who fetches a tool, uses it, and comes back. Ask "what's the weather?" and instead of guessing, it goes and checks a weather tool first.
中文解释：能在循环里采取行动、不只聊天的 LLM。
打个比方：普通 AI 只会说话；智能体还能做事，像一个会起身去拿工具、用完再回来的助手。你问"天气怎么样？"，它不瞎猜，而是先去查天气工具。
Tool calling / function calling（工具调用 / 函数调用） — the model asks to run a function; you run it and return the result.
Think of it like: Like a kid who can't reach the cookie jar asking a grown-up "please get the cookies." The AI can't check the weather itself, so it asks for get_weather("Paris"); you run it and hand back "18 degrees."
中文解释：模型请求运行某个函数，你运行后把结果交回。
打个比方：像够不到饼干罐的小孩对大人说"请帮我拿饼干"。AI 自己查不了天气，就请求工具 get_weather("Paris")；你运行后递回"18 度"，AI 接着往下做。
Tool schema（工具描述 / 模式） — the description of a tool's name, inputs, and purpose.
Think of it like: The instruction label on a tool. A juice machine labeled "put in fruit, get out juice." The schema tells the AI: this tool is get_weather and it needs a city name.
中文解释：描述工具名字、输入和用途的说明。
打个比方：工具上的说明标签。一台果汁机标着"放水果，出果汁"。schema 告诉 AI：这个工具叫 get_weather，需要一个城市名。
ReAct（ReAct / 边想边做） — a reason-then-act loop.
Think of it like: Think out loud, then act, then repeat. "I need the price (think) -> look it up (act) -> it's $20 (result) -> now I can answer." Like solving a treasure hunt one clue at a time.
中文解释："推理—行动"的循环。
打个比方：先想出声，再行动，再重复。"我需要价格（想）→ 我去查（做）→ 是 20 元（看结果）→ 现在能回答了。"像一步步解寻宝线索。
AgentExecutor (deprecated)（AgentExecutor（已弃用）） — LangChain's old agent runtime, replaced by create_agent.
Think of it like: This was the OLD toy for building agents. There's a newer, better toy now (create_agent). In an interview, show you know the new one is what people actually use today.
中文解释：LangChain 旧的智能体运行器，已被 create_agent 取代。
打个比方：这是搭智能体用的"旧玩具"。现在有更好的新玩具（create_agent）。面试时最好表现出你知道大家如今用的是新的那个。
Multi-agent / supervisor / handoff（多智能体 / 主管 / 交接） — several specialized agents working together.
Think of it like: A team of specialists. One "boss" helper hears your question and passes it to the right expert: math questions to the math helper, art to the art helper. They work as a team.
中文解释：多个各有专长的智能体协作。
打个比方：一队专家。一个"主管"助手听你的问题，再交给合适的专家：数学问题给数学助手，美术问题给美术助手。他们像团队一样配合。
Sub-agent（子智能体） — an agent another agent calls for a smaller task.
Think of it like: When one helper has a big job, it hires a little helper for one piece, like a chef asking a kitchen helper to just chop the onions.
中文解释：被另一个智能体调用去做小任务的智能体。
打个比方：一个助手活儿太大时，就雇个小助手干其中一块，像大厨让厨房小工只切洋葱，自己做别的。
Human-in-the-loop (HITL)（人在回路） — pause for a person's approval mid-task.
Think of it like: Like a kid who must ask a parent before spending allowance. The AI does most of the work but pauses to ask "is this okay?" before something big, like sending money.
中文解释：中途暂停、等人确认。
打个比方：像花零花钱前要先问家长的小孩。AI 大部分活自己干，但做大事前（比如转账）会停下来问人一句"这样行吗？"。
Memory (short / long-term)（记忆（短期 / 长期）） — remembering within a chat vs across chats.
Think of it like: Short-term memory is remembering what we said earlier today in this same chat. Long-term memory is remembering your name next week, like a friend who remembers you.
中文解释：同一次对话内记住 vs 跨对话记住。
打个比方：短期记忆是记得我们今天这次聊过什么。长期记忆是你下周再来，它还记得你的名字和最喜欢的颜色，像个记得你的朋友。
MCP (Model Context Protocol)（MCP / 模型上下文协议） — an open standard plug for connecting models to tools/data.
Think of it like: A universal plug, like a USB port. Instead of every tool needing its own connector, MCP is one standard plug so the AI can connect to email, calendars, and apps easily.
中文解释：连接模型与外部工具/数据的开放标准。
打个比方：一个通用插口，像 USB。不用每个工具都配专属接头，MCP 是统一插口，让 AI 轻松连上邮箱、日历和各种应用。
Guardrails（护栏） — rules that keep the agent's behavior safe.
Think of it like: Like the bumpers in bowling that keep the ball out of the gutter. Guardrails stop the AI from doing something it shouldn't, like saying mean words or deleting important files.
中文解释：约束智能体行为的规则。
打个比方：像保龄球道边上的挡板，别让球掉进沟里。护栏就是阻止 AI 做不该做的事的规则，比如说脏话或删掉重要文件。
LangChain Terms  LangChain 术语
LCEL (the "|" pipe)（LCEL（| 管道）） — composing components with the pipe operator.
Think of it like: A factory conveyor belt where each station does one job and passes it on. The "|" connects the stations: shape the question -> ask the model -> tidy the answer.
中文解释：用管道符把组件拼起来。
打个比方：一条工厂传送带，每个工位干一件事再传给下一个。"|"把工位连起来：整理问题 → 问模型 → 整理答案。
Runnable（Runnable / 可运行对象） — the shared .invoke / .stream / .batch interface.
Think of it like: A common "start button" that works the same on every machine. No matter which factory part it is, you start it the same way, so they all snap together neatly.
中文解释：通用的 .invoke / .stream / .batch 接口。
打个比方：一个在每台机器上都一样的"启动按钮"。不管是哪个工厂部件，启动方式都一样，于是它们能整齐地拼到一起。
Chain（链） — a straight line of steps, one after another.
Think of it like: Like following a recipe top to bottom: mix -> bake -> frost. Each step leads to the next, and you don't go backwards.
中文解释：一条直线式、一步接一步的流程。
打个比方：像从上到下照菜谱做：搅拌 → 烘烤 → 抹糖霜。每步带到下一步，不能往回走。
ChatPromptTemplate / MessagesPlaceholder（聊天提示模板 / 消息占位符） — a reusable prompt with a slot for history.
Think of it like: A fill-in-the-blanks form for talking to the AI. The placeholder is the blank where you slot in the conversation so far, so the AI remembers what was said.
中文解释：可复用的提示，带一个放历史的空位。
打个比方：一张和 AI 说话用的填空表。占位符就是那个空格，把"目前为止的对话"塞进去，AI 就记得之前说过什么。
Output parser（输出解析器） — turns the model's text into structured data.
Think of it like: A translator that turns the AI's words into something tidy a computer can use, like turning "yes!" into a checkmark (True).
中文解释：把模型的文字变成结构化数据。
打个比方：一个翻译员，把 AI 的话变成电脑好用的整齐形式，比如把"yes!"变成一个对勾（True）。
Document loader / splitter / retriever（文档加载器 / 分割器 / 检索器） — the RAG ingest-and-fetch parts.
Think of it like: The RAG kitchen crew: one loads the food (documents), one chops it into pieces (splitter), one fetches the right pieces when you're hungry (retriever).
中文解释：RAG 里负责装入和取出的部件。
打个比方：RAG 的厨房小队：一个把食材（文档）端进来，一个切成块（分割器），一个在你饿时取出正确的块（检索器）。
create_agent（create_agent） — the current high-level way to build a tool-using agent.
Think of it like: A ready-made kit for building a helper that can use tools. Instead of building from scratch, hand it a model and a box of tools and it's good to go.
中文解释：当前搭建工具型智能体的高层方式。
打个比方：一个现成的套件，用来搭能用工具的助手。不用从零做，把模型和一盒工具递给它，就能开工。
LangGraph Terms  LangGraph 术语
StateGraph（状态图） — the graph of nodes and edges you build.
Think of it like: A board-game map of "what happens next." Each space is a step, and arrows show which space you move to. You build this map to tell the AI how to flow through a task.
中文解释：你用节点和边搭起来的图。
打个比方：一张"接下来发生什么"的棋盘地图。每个格子是一步，箭头指向你接下来去哪格。你画这张图来告诉 AI 怎么走完一个任务。
State（状态） — the shared data every node reads and updates.
Think of it like: A shared backpack everyone on the trip can reach into and add to. It holds everything the task needs so far: the question, the notes found, the answer.
中文解释：每个节点都能读写的共享数据。
打个比方：一个旅途中大家都能伸手取放东西的共享背包。它装着任务目前需要的一切：问题、找到的笔记、答案。
Reducer (add_messages)（Reducer（add_messages）） — the rule for how updates merge into state.
Think of it like: A rule about how to put things in the backpack. "add_messages" means new notes get stacked on top instead of replacing the old ones, so the whole conversation builds up.
中文解释：规定状态更新如何合并的规则。
打个比方：关于怎么往背包里放东西的规则。"add_messages"表示新笔记是叠在上面，而不是把旧的扔掉替换，这样整段对话会越积越多。
Node（节点） — a function: state in, partial update out.
Think of it like: One stop on the map that does one job, like a kitchen station: it takes the backpack, does its task (e.g. "find the pages"), and puts the result back.
中文解释：一个函数：传入状态，返回部分更新。
打个比方：地图上干一件事的一个站点，像厨房工位：拿过背包，做它的活（比如"找到那些页"），再把结果放回去。
Edge / conditional edge（边 / 条件边） — a fixed step vs a step chosen by a check.
Think of it like: Arrows on the board game. A normal arrow always points to the next space. A conditional arrow is a fork: you go left or right depending on what just happened.
中文解释：固定的下一步 vs 由判断决定的下一步。
打个比方：棋盘上的箭头。普通箭头总是指向下一格。条件箭头是岔路口：根据刚刚发生的事，决定往左还是往右。
Cycle（环 / 循环） — a loop back to an earlier node.
Think of it like: A loop on the board that sends you back to try again, like "go back 3 spaces and re-roll." This is the special trick: a plain recipe can't loop, but this map can.
中文解释：回到前面某个节点的循环。
打个比方：棋盘上把你送回去再来一次的圈，像"后退 3 格、重掷骰子"。这是它的独门绝技：普通菜谱不能循环，这张图可以。
Checkpointer（检查点 / 存档器） — the persistence layer for state.
Think of it like: A save point in a video game. It remembers where you are so you can pick up later. In-memory saves only while the game is on; SQLite saves to a file so it survives a restart.
中文解释：状态的持久化层。
打个比方：电子游戏里的存档点，记住你走到哪，方便以后接着玩。内存存档只在游戏开着时有效；SQLite 存到文件里，关机后也还在。
Thread / thread_id（线程 / thread_id） — a label identifying one conversation.
Think of it like: A name tag on each conversation so they don't get mixed up, like labeling each kid's lunchbox. "user-123" keeps that person's chat separate.
中文解释：标识一次对话的标签。
打个比方：给每次对话贴的名牌，免得混在一起，像给每个孩子的午餐盒贴名字。"user-123"把那个人的聊天和别人分开。
Interrupt（中断） — pause the graph for human input.
Think of it like: A pause button that stops the game to ask a person something before continuing, like a "wait, check with a grown-up first" moment.
中文解释：暂停图、等待人输入。
打个比方：一个暂停按钮，停下来先问人一句再继续，像"等等，先问问大人"的那一刻。
Subgraph（子图） — a graph used as a node inside another graph.
Think of it like: A mini board-game tucked inside the big one. One space on the big map opens into its own little set of steps, then returns to the main board.
中文解释：当作一个节点用在另一张图里的图。
打个比方：大棋盘里塞着的一张小棋盘。大地图上的一个格子悄悄展开成它自己的一套小步骤，走完再回到主盘。
Streaming modes（流式模式） — ways to stream progress (values/updates/messages).
Think of it like: Watching the answer appear letter by letter as it's typed, instead of waiting for the whole thing, like seeing a text message being written in real time.
中文解释：流式输出进度的方式（values/updates/messages）。
打个比方：看着答案一个字一个字蹦出来，而不是等整段都好，像看一条短信被实时打出来。
Evaluation & Observability  评测与可观测性
Golden / ground-truth dataset（标准答案数据集） — known-correct Q&A to test against.
Think of it like: An answer key, like the back of the workbook. You compare the AI's answers to these known-correct ones to see how it's doing.
中文解释：用来对照的已知正确问答。
打个比方：一份答案册，像练习册背后的答案。你把 AI 的回答和这些已知正确答案对照，看它做得怎么样。
Retrieval metrics (recall@k, precision@k)（检索指标（recall@k 等）） — did you fetch the right chunks?
Think of it like: A score for "did you find the right book?" Recall@5 asks: was the correct book in the top 5 you grabbed? Higher means better finding.
中文解释：衡量"有没有找对块"。
打个比方：给"有没有找到对的书"打分。Recall@5 问：正确那本书在不在你拿回的前 5 本里？越高越会找。
Answer metrics (faithfulness, relevance)（答案指标（忠实度、相关性）） — is the answer grounded and on-topic?
Think of it like: A score for "is the answer actually good?" Faithfulness checks the answer matches the book (no making things up); relevance checks it actually answers the question asked.
中文解释：衡量"答案好不好、贴不贴题"。
打个比方：给"答案到底好不好"打分。忠实度检查答案是否符合书（没瞎编）；相关性检查它是否真的回答了被问的问题。
RAGAS（RAGAS） — a framework that auto-scores RAG quality.
Think of it like: A robot grader that automatically scores how good your AI's answers and searching are, so you don't check every one by hand.
中文解释：自动给 RAG 质量打分的框架。
打个比方：一个机器人阅卷员，自动给你 AI 的答案和检索打分，不用你一条条手动检查。
LLM-as-judge（用大模型当裁判） — using a model to grade outputs.
Think of it like: Using one AI as the teacher who grades another AI's homework, giving it a score like 1 to 5.
中文解释：用一个模型给输出打分。
打个比方：让一个 AI 当老师，给另一个 AI 的作业打分，比如打 1 到 5 分。
LangSmith（LangSmith） — tracing and observability for LLM apps.
Think of it like: A see-everything window. It shows each step the AI took, how long it took, and how much it cost, like instant replay for the AI's work.
中文解释：LLM 应用的追踪与可观测平台。
打个比方：一扇"全都看得见"的窗。它显示 AI 走的每一步、花了多久、花了多少钱，像 AI 工作的即时回放。
Regression / eval suite（回归测试 / 评测集） — automated tests run after changes.
Think of it like: A set of practice tests you re-run after any change, to make sure you didn't break something that used to work, like checking your bike still rides after fixing the seat.
中文解释：改动后自动跑的质量测试。
打个比方：一套练习测试，每次改动后都重跑一遍，确保没把原本好用的东西弄坏，就像修了车座后再确认自行车还能骑。
Backend, API & Systems  后端、API 与系统
REST API / endpoint（REST API / 接口端点） — an HTTP interface programs call.
Think of it like: A drive-thru window for computers. You pull up to a specific window (the endpoint), order in a standard way, and get something back. Programs talk through these windows.
中文解释：程序之间调用的 HTTP 接口。
打个比方：给电脑用的得来速窗口。你开到某个窗口（端点），用标准方式下单，再拿到东西。程序之间就通过这些窗口对话。
API key / bearer token（API 密钥 / 令牌） — the secret that authenticates your request.
Think of it like: A secret password that proves it's really you at the window. Keep it hidden (not written on your backpack), or someone could use your account and run up your bill.
中文解释：证明请求是你发的秘密凭证。
打个比方：一个证明"真的是你"的秘密口令。你要把它藏好（别写在书包上），否则别人会用你的账号、花你的钱。
Rate limiting / throttling（限流） — caps on how many requests you can send.
Think of it like: A rule like "only 3 cookies per kid." Ask for too many too fast and the window says "slow down!" That's the 429 message.
中文解释：对请求次数的上限。
打个比方：像"每个孩子只能拿 3 块饼干"的规矩。你要得太多太快，窗口就说"慢点！"——那就是 429 提示。
Retries / exponential backoff（重试 / 指数退避） — retry failures with growing wait times.
Think of it like: If the door is locked, you don't bang on it nonstop. You wait 1 second and try again, then 2, then 4, giving it room instead of spamming.
中文解释：失败后用越来越长的间隔重试。
打个比方：门锁着时，你不会一直猛敲。你等 1 秒再试，再等 2 秒，再等 4 秒，给它喘息，而不是狂刷。
Streaming (SSE / WebSockets)（流式（SSE / WebSocket）） — send the answer piece by piece.
Think of it like: Getting your answer word-by-word as it's made, like a teacher writing on the board live, instead of waiting for the whole essay.
中文解释：一点一点把答案发过来。
打个比方：答案边生成边一个词一个词到手，像老师在黑板上现场写，而不是等整篇作文写完。
Async / concurrency（异步 / 并发） — handle many requests at the same time.
Think of it like: Instead of making one sandwich, finishing, then starting the next, you start them all at once and they cook together, so nobody waits in a long line.
中文解释：同时处理很多请求。
打个比方：不是做完一个三明治再做下一个，而是一次全开火、一起做，这样没人排长队。
Status codes（状态码） — the result signal of a request.
Think of it like: Little signal lights from the window: green (200) = "all good!", 401 = "wrong password," 429 = "too fast, slow down," 500 = "the kitchen broke."
中文解释：请求结果的信号。
打个比方：窗口给的小信号灯：绿灯（200）="一切正常！"，401="口令错了"，429="太快，慢点"，500="后厨坏了"。
Pydantic（Pydantic） — Python data validation for inputs/outputs.
Think of it like: A bouncer that checks every order makes sense before letting it in. If a form needs a number but someone wrote "banana," it says "nope, fix it."
中文解释：Python 的数据校验。
打个比方：一个检查每张订单是否合理才放行的保安。表里要数字、有人却写了"香蕉"，它就说"不行，改一下"。
FastAPI / Flask（FastAPI / Flask） — Python frameworks to build your own API.
Think of it like: The tools you use to build your own drive-thru window, so other programs can place orders to your AI.
中文解释：用来搭建你自己 API 的 Python 框架。
打个比方：你用来搭自己得来速窗口的工具，好让别的程序来向你的 AI 下单。
Webhooks（Webhook / 回调） — the service calls you when a job is done.
Think of it like: Instead of you phoning the pizza place over and over asking "ready yet?", they call YOU when it's done. The service rings your doorbell when the job finishes.
中文解释：任务完成时由服务主动通知你。
打个比方：不用你反复打电话问比萨店"好了没？"，而是它好了主动打给你。服务在任务完成时来按你的门铃。
Caching / prompt caching（缓存 / 提示缓存） — reuse saved work to cut cost and time.
Think of it like: Keeping a snack you already made so you don't remake it. Ask the same big thing again and the AI reuses the saved work, faster and much cheaper.
中文解释：复用已算好的结果，省钱省时间。
打个比方：把已经做好的零食留着，省得重做。你再问同样的大问题，AI 复用存好的结果，更快也便宜得多。
Message queue / background jobs（消息队列 / 后台任务） — offload long tasks to a waiting line.
Think of it like: A waiting line for big chores. Instead of standing there, you drop your chore in the line, get told "we'll handle it," and go play while it runs.
中文解释：把耗时任务丢进排队线处理。
打个比方：给大活儿排的队。不让你干站着，你把活儿丢进队列，被告知"我们来处理"，然后你就去玩，它在后台跑。
Database (SQL vs NoSQL)（数据库（SQL / NoSQL）） — organized storage for data and history.
Think of it like: The big storage room where you keep things to remember: neat labeled shelves (SQL) or a flexible toy bin (NoSQL). With pgvector it can even store embeddings.
中文解释：存数据和历史的有序仓库。
打个比方：存东西、用来记住的大仓库：整齐贴标签的架子（SQL），或灵活的玩具箱（NoSQL）。用 pgvector 它还能存嵌入向量。
Docker / containerization（Docker / 容器化） — package the app to run anywhere.
Think of it like: Packing your whole setup into a lunchbox that works the same anywhere, your kitchen, grandma's kitchen, or school, so nothing's missing when you open it.
中文解释：把应用打包，到哪都能一样运行。
打个比方：把你整套配置装进一个饭盒，在哪打开都一样：你家厨房、奶奶家厨房、还是学校，打开都不缺东西。
Logging / monitoring（日志 / 监控） — record what happened and alert on problems.
Think of it like: A diary of everything that happened, plus an alarm that beeps when something goes wrong, so you can find out what broke and when.
中文解释：记录发生了什么，并在出问题时报警。
打个比方：一本记下所有事的日记，再加一个出事就响的警报，方便你查清什么时候、哪里坏了。
Idempotency（幂等性） — repeating a request causes no extra effect.
Think of it like: Pressing the elevator button 5 times doesn't call 5 elevators, it still comes once. Idempotency means sending the same request again doesn't do extra stuff, like charging you twice.
中文解释：同一请求重复发也不会产生额外效果。
打个比方：电梯按钮按 5 下不会来 5 部电梯，还是来一部。幂等就是同一个请求再发一次也不会多做事，比如不会重复扣两次钱。
Cost & Performance  成本与性能
Token cost / pricing（词元成本 / 计费） — you pay per input and output token.
Think of it like: You pay by the piece, like buying candy by weight. More words going in and out means more tokens, which means a bigger bill.
中文解释：按输入和输出的词元数收费。
打个比方：按量收费，像称重买糖。进出的字越多，词元越多，账单越大。
Prompt caching（提示缓存） — about 90% cheaper on repeated prompt prefixes.
Think of it like: Handing over the same big stack of papers every time is wasteful. Caching saves that stack so reusing it is almost free, around 90% cheaper.
中文解释：重复的提示前缀可便宜约 90%。
打个比方：每次都递同样一大叠纸太浪费。缓存把这叠纸存起来，复用时几乎免费，便宜约九成。
Batch processing（批处理） — about 50% cheaper for non-urgent bulk jobs.
Think of it like: Doing a giant pile of homework all at once overnight, instead of one sheet at a time, gets a half-price discount. Great when you're not in a hurry.
中文解释：不急的大批量任务便宜约 50%。
打个比方：一大堆作业不一张张做，而是放到晚上一次性做完，能打对折。不赶时间时很划算。
Latency vs throughput（延迟 vs 吞吐量） — speed for one request vs requests per second.
Think of it like: Latency is how long ONE kid waits for lunch. Throughput is how many kids the cafeteria feeds per minute. You can be quick for one and still feed a big crowd.
中文解释：单个请求的快慢 vs 每秒能处理多少。
打个比方：延迟是一个孩子等午饭等多久；吞吐量是食堂每分钟能喂多少孩子。可以对一个人很快，同时还喂得了一大群。
Context-window management（上下文窗口管理） — trim or summarize to fit and save cost.
Think of it like: Since the AI's "desk" is small, keep it tidy: summarize old notes into a short version instead of piling on every paper, leaving room for what matters now.
中文解释：修剪或摘要以装得下、又省钱。
打个比方：AI 的"桌子"小，所以保持整洁：把旧笔记总结成简短版，而不是把每张纸都堆上去，给真正要紧的留出空间。

