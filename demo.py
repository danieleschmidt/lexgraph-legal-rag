#!/usr/bin/env python3
"""
demo.py — LexGraph Legal RAG demo

Indexes 3 sample legal documents, runs 3 queries, shows ranked results with citations.
No external API calls — fully offline using FAISS + TF-IDF.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lexgraph_legal_rag.indexer import LegalDocumentIndexer
from lexgraph_legal_rag.retriever import LegalRetriever, MultiAgentRetriever


# ── Sample legal documents ──────────────────────────────────────────────────

SAMPLE_DOCUMENTS = {
    "NDA_Agreement_2024": """
MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of January 1, 2024,
by and between Acme Corp ("Disclosing Party") and Beta LLC ("Receiving Party").

Section 1. Definition of Confidential Information.
"Confidential Information" means any and all technical and non-technical information,
including but not limited to patent, trade secret, proprietary information, techniques,
sketches, drawings, models, inventions, know-how, processes, apparatus, and formulae.

Section 2. Obligations of Receiving Party.
The Receiving Party agrees to hold the Confidential Information in strict confidence.
The Receiving Party shall not disclose any Confidential Information to third parties
without the prior written consent of the Disclosing Party.
The Receiving Party shall use the Confidential Information solely for the purpose of
evaluating a potential business relationship between the parties.

Section 3. Term and Termination.
This Agreement shall remain in effect for a period of three (3) years from the date
of execution, unless earlier terminated by either party upon thirty (30) days written notice.
Upon termination, the Receiving Party shall promptly return or destroy all Confidential Information.

Section 4. Governing Law.
This Agreement shall be governed by and construed in accordance with the laws of the
State of Delaware, without regard to its conflict of law provisions.

Section 5. Remedies.
The parties acknowledge that breach of this Agreement may cause irreparable harm for which
monetary damages would be inadequate. Therefore, the parties agree that injunctive relief
is an appropriate remedy for any breach.
""",

    "Software_License_Agreement": """
SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("License") is made effective as of March 15, 2024,
between TechCorp Inc. ("Licensor") and Enterprise Solutions Ltd. ("Licensee").

Article 1. Grant of License.
Licensor hereby grants to Licensee a non-exclusive, non-transferable, limited license
to use the Software solely for Licensee's internal business purposes.
The license does not include the right to sublicense, modify, or distribute the Software.

Article 2. Restrictions.
Licensee shall not:
(a) copy or duplicate the Software except for backup purposes;
(b) sell, transfer, assign, or sublicense the Software or any rights therein;
(c) remove or alter any proprietary notices, labels, or marks on the Software;
(d) reverse engineer, decompile, or disassemble the Software.

Article 3. Intellectual Property.
All intellectual property rights in and to the Software, including all modifications
and enhancements, shall remain the exclusive property of the Licensor.
Licensee acknowledges that no title or ownership in the Software is transferred.

Article 4. Warranty Disclaimer.
THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE, AND NONINFRINGEMENT.

Article 5. Limitation of Liability.
IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL
DAMAGES ARISING OUT OF OR RELATED TO THIS AGREEMENT OR THE USE OF THE SOFTWARE.
LICENSOR'S TOTAL LIABILITY SHALL NOT EXCEED THE FEES PAID BY LICENSEE IN THE TWELVE MONTHS
PRECEDING THE CLAIM.

Article 6. Term and Termination.
This License is effective until terminated. Licensor may terminate this License immediately
upon written notice if Licensee breaches any provision of this Agreement.
Upon termination, Licensee shall cease all use of the Software and destroy all copies.
""",

    "Employment_Contract_Template": """
EMPLOYMENT CONTRACT

This Employment Contract ("Contract") is entered into between Globaltech Corp ("Employer")
and the individual named herein ("Employee"), effective as of the start date specified below.

Clause 1. Position and Duties.
Employee is hired for the position of Senior Software Engineer.
Employee shall perform all duties as assigned by the Employer and shall devote full
working time and attention to the business of the Employer.

Clause 2. Compensation.
Employer shall pay Employee an annual base salary of $120,000 USD, payable bi-weekly.
Employee shall be eligible for an annual performance bonus at the sole discretion of Employer.
Employee benefits include health insurance, dental coverage, and a 401(k) plan.

Clause 3. Confidentiality and Non-Disclosure.
Employee acknowledges that during the course of employment, Employee will have access to
trade secrets and confidential business information of Employer.
Employee agrees to maintain the confidentiality of all such information during and after
the term of employment. This obligation survives termination of employment.

Clause 4. Intellectual Property Assignment.
All inventions, developments, improvements, and works of authorship created by Employee
during employment, whether or not during working hours, that relate to the business of
Employer, shall be the exclusive property of Employer.
Employee hereby assigns to Employer all rights, title, and interest in such intellectual property.

Clause 5. Non-Compete.
During the term of employment and for a period of one (1) year following termination,
Employee shall not engage in any business activity that directly competes with Employer
within the geographic area of the United States.

Clause 6. Termination.
Either party may terminate this Contract upon thirty (30) days written notice.
Employer may terminate this Contract immediately for cause, including but not limited to:
misconduct, breach of this Contract, or failure to perform duties.
Upon termination, all Confidential Information and property of Employer shall be returned.

Clause 7. Governing Law and Dispute Resolution.
This Contract shall be governed by the laws of California.
Any disputes arising from this Contract shall be resolved through binding arbitration
in San Francisco, California, under the rules of the American Arbitration Association.
""",
}

QUERIES = [
    "What are the confidentiality obligations for the receiving party?",
    "What happens to intellectual property created during employment?",
    "What are the limitations on software liability and warranty?",
]


def run_demo():
    print("=" * 70)
    print("  LexGraph Legal RAG — Document Retrieval Demo")
    print("  FAISS + TF-IDF (offline, no API calls)")
    print("=" * 70)
    print()

    # ── Index documents ──────────────────────────────────────────────────────
    print(">>> Indexing documents...")
    indexer = LegalDocumentIndexer(chunk_size=400, chunk_overlap=50)
    total_chunks = indexer.index_documents(SAMPLE_DOCUMENTS)
    print(f"    Indexed {len(SAMPLE_DOCUMENTS)} documents → {total_chunks} chunks")
    print()

    # ── Single-agent retriever ────────────────────────────────────────────────
    retriever = LegalRetriever(indexer=indexer, top_k=3)
    multi_retriever = MultiAgentRetriever(indexer=indexer, top_k=3)

    for i, query in enumerate(QUERIES, 1):
        print(f"Query {i}: {query}")
        print("-" * 70)

        # Standard retrieval
        print("  [LegalRetriever — top 3 results]")
        results = retriever.retrieve(query)
        for j, r in enumerate(results, 1):
            print(f"  {j}. Score: {r.score:.4f} | {r.citation}")
            snippet = r.text[:200].replace("\n", " ").strip()
            print(f"     {snippet}{'...' if len(r.text) > 200 else ''}")
        print()

        # Multi-agent retrieval
        print("  [MultiAgentRetriever — expanded queries, deduplicated]")
        ma_results = multi_retriever.retrieve(query)
        for j, r in enumerate(ma_results[:5], 1):
            print(f"  {j}. Score: {r.score:.4f} | {r.citation}")
        print()
        print()

    print("=" * 70)
    print("  Demo complete. Index ready for DocGraph integration.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
