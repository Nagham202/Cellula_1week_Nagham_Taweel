import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FOLDER = Path(__file__).resolve().parent

with open(FOLDER / "results.json") as f:
    results = json.load(f)
test = results["test"]
per_class = results["test_per_class"]

styles = getSampleStyleSheet()
styles["Heading2"].keepWithNext = 1
grid = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, "grey"),
                   ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")])
story = []

story.append(Paragraph("LSTM - Training Results", styles["Title"]))
story.append(Paragraph("Summary", styles["Heading2"]))
story.append(Paragraph(
    "An LSTM was trained to classify text into 5 toxic categories using toxic_data.csv, with the same data "
    "preparation as the RNN: each input is the query joined with its image description, duplicate rows were "
    "removed before splitting the data into 70% train, 15% validation and 15% test, and 4 classes with only "
    "2-5 unique examples were excluded. The model uses 64 LSTM units, dropout 0.3, class weights and early "
    f"stopping. On the test set it reached a weighted F1 of <b>{test['weighted_f1']:.3f}</b>, meeting the "
    f"F1 &gt;= 0.85 requirement, and a macro F1 of <b>{test['macro_f1']:.3f}</b>, which is below it. "
    "The gap comes from one class: Unknown S-Type (13 test rows) shares its image descriptions with Safe, "
    "so the model often confuses the two. The other four classes score 0.92-1.00.", styles["BodyText"]))

story.append(Paragraph("Test results", styles["Heading2"]))
overall = Table([
    ["", "Macro", "Weighted"],
    ["F1", f"{test['macro_f1']:.3f}", f"{test['weighted_f1']:.3f}"],
    ["Precision", f"{test['macro_precision']:.3f}", f"{test['weighted_precision']:.3f}"],
    ["Recall", f"{test['macro_recall']:.3f}", f"{test['weighted_recall']:.3f}"],
], hAlign="LEFT")
overall.setStyle(grid)
story.append(overall)
story.append(Spacer(1, 8))

rows = [["Class", "Precision", "Recall", "F1", "Test rows"]]
for name in results["data"]["classes"]:
    c = per_class[name]
    rows.append([name, f"{c['precision']:.3f}", f"{c['recall']:.3f}", f"{c['f1-score']:.3f}", int(c["support"])])
classes = Table(rows, hAlign="LEFT")
classes.setStyle(grid)
story.append(classes)

story.append(Paragraph("Training and validation curves", styles["Heading2"]))
story.append(Image(str(FOLDER / "training_curves.png"), width=17 * cm, height=6.2 * cm))
story.append(Paragraph("Confusion matrix (test set)", styles["Heading2"]))
story.append(Image(str(FOLDER / "confusion_matrix.png"), width=13 * cm, height=11.2 * cm))

SimpleDocTemplate(str(FOLDER / "LSTM_report.pdf"), pagesize=A4,
                  topMargin=1.5 * cm, bottomMargin=1.5 * cm).build(story)
print("Saved", FOLDER / "LSTM_report.pdf")
