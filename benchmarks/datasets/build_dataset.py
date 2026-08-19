"""Dataset builder for HRA-RLM benchmarks.

Why this module exists:
Creates a small, controlled test set of documents and Q/A pairs for benchmarking.
Mirrors the MIT paper's S-NIAH / OOLONG-style tasks with factual-lookup and multi-hop questions.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Sample documents (public-domain texts)
DOCUMENTS = [
    {
        "id": "doc1",
        "text": """The theory of relativity is a scientific theory developed by Albert Einstein. It consists of two parts:
special relativity and general relativity. Special relativity applies to all physical phenomena in the absence of gravity.
General relativity explains the law of gravitation and its relation to other forces of nature. The theory has been confirmed
by many experiments and observations, including the bending of light by the sun, the precession of Mercury's orbit,
and the gravitational redshift of light.""",
    },
    {
        "id": "doc2",
        "text": """Photosynthesis is the process used by plants, algae, and certain bacteria to convert light energy into
chemical energy. It occurs in chloroplasts, which contain the pigment chlorophyll. The overall reaction is:
6CO2 + 6H2O -> C6H12O6 + 6O2. This process is essential for life on Earth as it produces oxygen and carbohydrates.
There are two stages: light-dependent reactions and the Calvin cycle. The light-dependent reactions occur in the thylakoid
membranes, while the Calvin cycle occurs in the stroma.""",
    },
    {
        "id": "doc3",
        "text": """The Renaissance was a period in European history marking the transition from the Middle Ages to modernity.
It began in Italy in the 14th century and spread to the rest of Europe by the 16th century. The Renaissance was characterized
by a revival of interest in classical art, literature, and learning. Key figures include Leonardo da Vinci, Michelangelo,
and Raphael. The invention of the printing press by Johannes Gutenberg around 1440 greatly facilitated the spread of
Renaissance ideas.""",
    },
    {
        "id": "doc4",
        "text": """The human genome is the complete set of nucleic acid sequences for humans, encoded as DNA within the
23 chromosome pairs in cell nuclei. The Human Genome Project, completed in 2003, determined the sequence of the
3 billion base pairs. Genes are segments of DNA that code for proteins. The genome contains approximately 20,000-25,000
protein-coding genes. Genetic variation, such as single nucleotide polymorphisms (SNPs), contributes to individual differences
and disease susceptibility.""",
    },
    {
        "id": "doc5",
        "text": """The Great Wall of China is a series of fortifications built along the northern borders of China to protect
against invasions. The most famous sections were built during the Ming dynasty (1368-1644). The wall stretches over
13,000 miles, including branches. It is made of stone, brick, tamped earth, wood, and other materials. Contrary to
popular belief, the wall is not a single continuous line but consists of many segments. It is one of the world's most
impressive architectural feats.""",
    },
]

# Generate Q/A pairs for each document
def generate_qa_pairs(doc_text: str, doc_id: str) -> List[Dict[str, str]]:
    """Generate a set of question-answer pairs for a document."""
    # This is a simplified version; in practice, you'd use an LLM to generate these.
    # For demo, we create hand-crafted pairs.
    # Real benchmark would use more diverse questions.
    pairs = {
        "doc1": [
            {"question": "Who developed the theory of relativity?", "answer": "Albert Einstein"},
            {"question": "What are the two parts of the theory of relativity?", "answer": "special relativity and general relativity"},
            {"question": "What phenomenon confirms general relativity?", "answer": "bending of light by the sun"},
        ],
        "doc2": [
            {"question": "What is the process by which plants convert light energy into chemical energy?", "answer": "photosynthesis"},
            {"question": "Where does photosynthesis occur in plants?", "answer": "chloroplasts"},
            {"question": "What is the overall reaction of photosynthesis?", "answer": "6CO2 + 6H2O -> C6H12O6 + 6O2"},
        ],
        "doc3": [
            {"question": "When did the Renaissance begin?", "answer": "14th century"},
            {"question": "Where did the Renaissance begin?", "answer": "Italy"},
            {"question": "Who is a key figure of the Renaissance?", "answer": "Leonardo da Vinci"},
        ],
        "doc4": [
            {"question": "How many base pairs does the human genome contain?", "answer": "3 billion"},
            {"question": "When was the Human Genome Project completed?", "answer": "2003"},
            {"question": "What are SNPs?", "answer": "single nucleotide polymorphisms"},
        ],
        "doc5": [
            {"question": "What is the approximate length of the Great Wall of China?", "answer": "13,000 miles"},
            {"question": "Which dynasty built the most famous sections?", "answer": "Ming dynasty"},
            {"question": "Is the Great Wall a single continuous line?", "answer": "No, it consists of many segments."},
        ],
    }
    return pairs.get(doc_id, [])


def build_dataset(output_dir: Path) -> None:
    """Build the benchmark dataset and save to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    for doc in DOCUMENTS:
        doc_id = doc["id"]
        text = doc["text"]
        qa_pairs = generate_qa_pairs(text, doc_id)
        dataset.append({
            "doc_id": doc_id,
            "text": text,
            "qa_pairs": qa_pairs,
        })

    # Save as JSON
    with open(output_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset built with {len(dataset)} documents and {sum(len(d['qa_pairs']) for d in dataset)} Q/A pairs.")
    print(f"Saved to {output_dir / 'dataset.json'}")


if __name__ == "__main__":
    from pathlib import Path
    build_dataset(Path(__file__).parent)