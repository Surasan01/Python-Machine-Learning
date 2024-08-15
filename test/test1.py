import spacy

# โหลดโมเดลภาษาอังกฤษ
nlp = spacy.load("en_core_web_sm")

# ข้อความตัวอย่าง
text = "Apple is looking at buying U.K. startup for $1 billion."

# การประมวลผลข้อความ
doc = nlp(text)

# แสดง Named Entities
for ent in doc.ents:
    print(ent.text, ent.label_)