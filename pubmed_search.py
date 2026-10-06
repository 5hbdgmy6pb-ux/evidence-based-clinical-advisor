import re

from pymedx import PubMed

# NCBI 要求提供一个联系邮箱来标识请求来源，建议换成你自己的邮箱。
CONTACT_EMAIL = "your_email@example.com"

pubmed = PubMed(tool="MyHealthAgent", email=CONTACT_EMAIL)


def extract_year(pub_date):
    """
    取出出版年份。

    pymedx 的 publication_date 有时是 date 对象，有时直接是字符串
    （如 "2023" / "2023 May" / "2023-05-01"）。原先直接 .year 会在
    字符串上抛 AttributeError，导致整条检索中断。两种形态都要兼容。
    """
    if not pub_date:
        return None
    year = getattr(pub_date, "year", None)
    if year:
        return year
    match = re.search(r"\b(?:19|20)\d{2}\b", str(pub_date))
    return int(match.group(0)) if match else None


def search_medical_literature(query, max_results=10):
    """
    从 PubMed 检索医学文献。
    优先返回系统评价、Meta 分析、RCT、综述和临床指南（证据等级更高）。
    """
    # 在查询中附加文章类型过滤，提高证据等级。
    # 除原始研究外还纳入综述与临床指南：诊断与治疗方案这类问题，
    # 指南（practice guideline）和综述（review）比单篇 RCT 更直接可用。
    enhanced_query = (
        f"({query}) AND "
        f"(systematic[sb] OR meta-analysis[pt] OR randomized controlled trial[pt] "
        f"OR review[pt] OR practice guideline[pt])"
    )

    results = pubmed.query(enhanced_query, max_results=max_results)

    articles = []
    for article in results:
        pub_date = getattr(article, "publication_date", None)
        articles.append({
            "title": getattr(article, "title", ""),
            "abstract": getattr(article, "abstract", "") or "",
            "authors": getattr(article, "authors", []),
            "journal": getattr(article, "journal", ""),
            "year": extract_year(pub_date),
            "doi": getattr(article, "doi", None),
            "pmid": getattr(article, "pubmed_id", None) or getattr(article, "pmid", None),
        })

    return articles


if __name__ == "__main__":
    results = search_medical_literature("metformin cardiovascular benefits", max_results=5)
    print(f"检索到 {len(results)} 篇文献：\n")
    for r in results[:3]:
        print(f"标题: {r['title']}")
        print(f"期刊: {r['journal']} ({r['year']})")
        print(f"摘要: {(r['abstract'] or '')[:200]}...")
        print("---")
