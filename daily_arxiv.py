import os
import json
import urllib.request
import fitz  # PyMuPDF，用于解析 PDF
import arxiv
from openai import OpenAI
import requests

HISTORY_FILE = "pushed_papers.json"

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://api.deepseek.com"  # 如果使用 OpenAI 或其他 API，修改此地址
)

def load_pushed_ids():
    """读取历史已推送的论文 ID 列表"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()

def save_pushed_ids(pushed_ids):
    """保存已推送的论文 ID 列表到文件"""
    recent_ids = list(pushed_ids)[-1000:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(recent_ids, f, ensure_ascii=False, indent=2)

def extract_pdf_content(pdf_url, max_pages=3):
    """下载 PDF 并提取前 N 页文本（包含摘要、引言与方法概述）"""
    try:
        req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            pdf_data = response.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        
        text = ""
        for page_num in range(min(max_pages, len(doc))):
            text += doc[page_num].get_text()
        return text[:4000]  # 限制文本长度，避免超出 Token
    except Exception as e:
        print(f"提取 PDF 失败 ({pdf_url}): {e}")
        return ""

def fetch_latest_papers(topic="cs.CV", max_results=15):
    """抓取最新论文并提取 PDF 详细内容"""
    search = arxiv.Search(
        query=f"cat:{topic}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    print("开始获取论文列表及深度 PDF 内容...")
    for result in arxiv.Client().results(search):
        paper_id = result.entry_id.split('/')[-1].split('v')[0]
        
        # 提取 PDF 前几页详细文本
        detail_text = extract_pdf_content(result.pdf_url, max_pages=3)
        
        papers.append({
            "id": paper_id,
            "title": result.title,
            "authors": [a.name for a in result.authors[:3]],
            "abstract": result.summary.replace("\n", " "),
            "detail_text": detail_text if detail_text else result.summary,
            "url": result.entry_id,
            "pdf": result.pdf_url
        })
    return papers

def analyze_paper_deeply(paper, keywords):
    """对单篇论文进行深度结构化分析（提取核心创新点与算法 Pipeline）"""
    prompt = f"""
你是一名顶级 AI 领域专家。请根据以下论文的内容（包含摘要、引言和方法部分），进行深度技术剖析。

【论文标题】：{paper['title']}
【关注领域关键词】：{keywords}
【论文详细文本】：
{paper['detail_text']}

请按以下固定格式输出该论文的深度解读（格式要求为 Markdown）：

### 📄 [{paper['title']}]({paper['url']})
- **作者**: {', '.join(paper['authors'])} | [PDF直达]({paper['pdf']})

💡 **核心创新点 (Key Innovations)**:
- 1. ... (解决了什么痛点/提出的核心理念)
- 2. ...

⚙️ **算法 Pipeline 拆解**:
- **Step 1 (输入/预处理)**: ...
- **Step 2 (核心模块/特征提取)**: ...
- **Step 3 (损失函数/优化目标)**: ...

🏗️ **网络架构/逻辑流程图**:
*(注：请根据论文真实的架构用字符画出具体的模块流向)*

---
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"分析论文失败 ({paper['id']}): {e}")
        return None

def send_to_feishu(webhook_url, content):
    data = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": "🔥 深度 arXiv 算法与架构精读推送"}},
            "elements": [{"tag": "markdown", "content": content}]
        }
    }
    resp = requests.post(webhook_url, json=data)
    return resp.status_code == 200

if __name__ == "__main__":
    TOPIC = "cs.CV"
    KEYWORDS = "pretrain, vla, world model, diffusion"
    
    pushed_ids = load_pushed_ids()
    raw_papers = fetch_latest_papers(topic=TOPIC, max_results=15)
    
    # 过滤去重
    new_papers = [p for p in raw_papers if p["id"] not in pushed_ids]
    print(f"共发现 {len(new_papers)} 篇未推送论文。")
    
    if not new_papers:
        print("今日无新论文。")
        exit(0)
    
    # 限制每日精读最多 5~10 篇，防止卡片文本过长
    selected_papers = new_papers[:5]
    
    full_report = ""
    for idx, paper in enumerate(selected_papers, 1):
        print(f"正在深度分析第 {idx}/{len(selected_papers)} 篇论文: {paper['title']}...")
        analysis = analyze_paper_deeply(paper, KEYWORDS)
        if analysis:
            full_report += analysis + "\n\n"
    
    feishu_webhook = os.getenv("FEISHU_WEBHOOK")
    if feishu_webhook and full_report:
        if send_to_feishu(feishu_webhook, full_report):
            print("推送成功！保存历史记录...")
            for p in selected_papers:
                pushed_ids.add(p["id"])
            save_pushed_ids(pushed_ids)
        else:
            print("推送失败。")
    else:
        print(full_report)