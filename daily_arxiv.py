import os
import arxiv
from openai import OpenAI
import requests

# 1. 初始化 AI 客户端（以 DeepSeek 或 OpenAI 为例）
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://api.deepseek.com"  # 可替换为标准的 OpenAI 或其他 API 地址
)

def fetch_latest_papers(topic="cs.CV", max_results=20):
    """从 arXiv 获取指定领域的最新论文"""
    search = arxiv.Search(
        query=f"cat:{topic}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for result in arxiv.Client().results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors[:3]],
            "abstract": result.summary.replace("\n", " "),
            "url": result.entry_id,
            "pdf": result.pdf_url
        })
    return papers

def summarize_and_filter(papers, keywords):
    """调用大模型进行智能筛选与总结"""
    prompt = f"""
你是一名严谨的研究员。以下是今日 arXiv 最新的论文列表。
你的关注领域关键词是：{keywords}。

请按以下要求处理：
1. 从中挑选出最符合关注领域的 10 篇论文。
2. 对每篇论文用中文生成简短介绍（包括：核心痛点、创新方法、实验效果/结论）。

论文数据如下：
{papers}

输出格式要求为 Markdown：
### [论文标题](URL)
- **作者**: ...
- **核心要点**: ...
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

def send_to_feishu(webhook_url, content):
    """将结果推送至飞书机器人（也可替换为 Telegram/钉钉/微信）"""
    data = {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": "📅 每日 arXiv 关注论文推送"}},
            "elements": [{"tag": "markdown", "content": content}]
        }
    }
    requests.post(webhook_url, json=data)

if __name__ == "__main__":
    # 配置你的领域关键词与分类
    TOPIC = "cs.CV" # 例如计算机视觉
    KEYWORDS = "pretrain, vla, world model, Diffusion"
    
    papers = fetch_latest_papers(topic=TOPIC, max_results=25)
    summary_report = summarize_and_filter(papers, KEYWORDS)
    
    # 推送给机器人
    feishu_webhook = os.getenv("FEISHU_WEBHOOK")
    if feishu_webhook:
        send_to_feishu(feishu_webhook, summary_report)
    else:
        print(summary_report)
