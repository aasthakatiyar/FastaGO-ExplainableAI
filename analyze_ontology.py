from utils import load_go_metadata
from collections import Counter
import pandas as pd
import os

def analyze_go_obo(path):
    print(f"Analyzing {path}...")
    meta = load_go_metadata(path)
    total = len(meta)
    print(f"Total terms parsed: {total}")
    
    # 1. Namespace distribution
    namespaces = [term.get("namespace") for term in meta.values()]
    ns_counts = Counter(namespaces)
    
    # 2. Status distribution
    obsolete_count = sum(1 for term in meta.values() if term.get("is_obsolete"))
    
    # 3. Field presence
    field_counts = {}
    for term in meta.values():
        for field, value in term.items():
            if value:
                if isinstance(value, (list, dict)):
                    if len(value) > 0:
                        field_counts[field] = field_counts.get(field, 0) + 1
                else:
                    field_counts[field] = field_counts.get(field, 0) + 1
                    
    # 4. Cross-references analysis
    xrefs = []
    for term in meta.values():
        for x in term.get("xref", []):
            if ":" in x:
                xrefs.append(x.split(":", 1)[0])
    xref_counts = Counter(xrefs)
    
    # Generate Report
    report = []
    report.append("# Gene Ontology (GO.obo) Analysis Report")
    report.append(f"\n**Total Terms:** {total}")
    report.append(f"\n**Active Terms:** {total - obsolete_count}")
    report.append(f"**Obsolete Terms:** {obsolete_count} ({obsolete_count/total*100:.1f}%)")
    
    report.append("\n## Namespace Distribution")
    for ns, count in ns_counts.most_common():
        report.append(f"- **{ns}**: {count} ({count/total*100:.1f}%)")
        
    report.append("\n## Field Presence Statistics")
    report.append("| Field | Count | Percentage |")
    report.append("|-------|-------|------------|")
    for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"| {field} | {count} | {count/total*100:.1f}% |")
        
    report.append("\n## Top 10 External Databases (xrefs)")
    for db, count in xref_counts.most_common(10):
        report.append(f"- **{db}**: {count} references")
        
    report_text = "\n".join(report)
    
    with open("GO_ANALYSIS_REPORT.md", "w") as f:
        f.write(report_text)
        
    print(f"\nAnalysis complete! Report saved to GO_ANALYSIS_REPORT.md")
    print(report_text)

if __name__ == "__main__":
    analyze_go_obo("data/go.obo")
