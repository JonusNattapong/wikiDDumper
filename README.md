# 🌟 WikiDumper - เครื่องมือสกัดข้อมูลวิกิพีเดีย

> เครื่องมือที่ทันสมัยและใช้งานง่ายสำหรับดาวน์โหลดและสกัดข้อความจาก Wikipedia dump files 
> รองรับ **Python 3.13** และออกแบบมาเพื่อการใช้งานกับภาษาไทยโดยเฉพาะ

![Python](https://img.shields.io/badge/Python-3.7%2B-blue) 
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 ภาพรวม

WikiDumper เป็นเครื่องมือที่พัฒนาขึ้นเพื่อแก้ปัญหาความเข้ากันได้ของ WikiExtractor เดิมกับ Python รุ่นใหม่ 
โดยมุ่งเน้นให้การสกัดข้อมูลจาก Wikipedia เป็นเรื่องง่ายและรวดเร็ว

### ✨ จุดเด่น

| คุณสมบัติ | รายละเอียด |
|-----------|------------|
| 🚀 **ทันสมัย** | รองรับ Python 3.13 และใช้ mwxml library ที่มีประสิทธิภาพ |
| 🇹🇭 **ภาษาไทย** | ปรับแต่งการทำความสะอาดข้อความสำหรับภาษาไทยโดยเฉพาะ |
| 🧹 **สะอาด** | ลบ wiki markup, template, references อัตโนมัติ |
| 📂 **จัดระเบียบ** | แบ่งผลลัพธ์เป็นไฟล์ย่อยขนาดเหมาะสม |
| ⚡ **รวดเร็ว** | ประมวลผลเร็วกว่า WikiExtractor เดิม |

---

## 🔧 การเตรียมพร้อม

### ความต้องการของระบบ

```
✅ Python 3.7+ (แนะนำ 3.9 หรือใหม่กว่า)
✅ Git (สำหรับ clone repository)  
✅ อินเทอร์เน็ต (สำหรับดาวน์โหลด dump files)
✅ พื้นที่ว่าง 2-5 GB (ขึ้นอยู่กับภาษา)
```

### การติดตั้ง

**1. Clone โปรเจกต์**

```bash
git clone https://github.com/your-username/wikiDDumper.git
cd wikiDDumper
```

**2. ติดตั้ง Dependencies**

```bash
pip install mwxml
```

## 🚀 คู่มือการใช้งาน

### 📥 ขั้นตอนที่ 1: ดาวน์โหลด Wikipedia Dump File

#### สำหรับ Wikipedia ภาษาไทย (แนะนำ):

```bash
curl -L -o thwiki-20250601-pages-articles-multistream.xml.bz2 \
  https://dumps.wikimedia.org/thwiki/20250601/thwiki-20250601-pages-articles-multistream.xml.bz2
```

#### สำหรับภาษาอื่น ๆ:

เยี่ยมชม [dumps.wikimedia.org](https://dumps.wikimedia.org/) และเลือก dump file ที่ต้องการ

**ตัวอย่างภาษายอดนิยม:**

```bash
# ภาษาอังกฤษ (ขนาดใหญ่มาก ~20GB)
curl -L -o enwiki-latest-pages-articles.xml.bz2 \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# ภาษาญี่ปุ่น
curl -L -o jawiki-latest-pages-articles.xml.bz2 \
  https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2
```

### ⚙️ ขั้นตอนที่ 2: สกัดข้อความจาก Dump File

#### รูปแบบคำสั่งพื้นฐาน:

```bash
python extract_wiki.py <input_dump_file> <output_directory>
```

#### ตัวอย่างการใช้งาน:

```bash
python extract_wiki.py thwiki-20250601-pages-articles-multistream.xml.bz2 output_dir
```

### 📊 พารามิเตอร์และการปรับแต่ง

| พารามิเตอร์ | ความหมาย | ตัวอย่าง |
|-------------|-----------|----------|
| `<input_dump_file>` | ไฟล์ Wikipedia dump (.xml.bz2) | `thwiki-20250601-pages-articles-multistream.xml.bz2` |
| `<output_directory>` | โฟลเดอร์เก็บผลลัพธ์ | `output_dir`, `thai_articles` |

#### การปรับแต่งภายในไฟล์ `extract_wiki.py`:

```python
articles_per_file = 100        # จำนวนบทความต่อไฟล์ (ค่าเริ่มต้น: 100)
min_text_length = 100          # ความยาวข้อความขั้นต่ำ (ค่าเริ่มต้น: 100 ตัวอักษร)
```

## 📁 ผลลัพธ์ที่ได้

### โครงสร้างไฟล์:

เครื่องมือจะสร้างไฟล์หลายไฟล์ในโฟลเดอร์ output:

- `wiki_00`, `wiki_01`, `wiki_02`, ...
- แต่ละไฟล์มีบทความประมาณ 100 บทความ
- ข้อความถูกทำความสะอาดแล้ว (ลบ wiki markup, template, references)

### รูปแบบเนื้อหา:

```xml
<doc id="123" title="พระบาทสมเด็จพระเจ้าอยู่หัว">
พระบาทสมเด็จพระเจ้าอยู่หัวภูมิพลอดุลยเดช มหิตลาธิเบศรราชัน 
พระราชทานนามว่า "พระบาทสมเด็จพระเจ้าอยู่หัวภูมิพลอดุลยเดช"...
</doc>

<doc id="456" title="กรุงเทพมหานคร">
กรุงเทพมหานคร เป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย...
</doc>
```

## ตัวอย่างการใช้งาน

### ดาวน์โหลดและสกัด Wikipedia ภาษาไทย
```bash
# ดาวน์โหลด dump file
curl -L -o thwiki-latest.xml.bz2 https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2

# สกัดข้อความ
python extract_wiki.py thwiki-latest.xml.bz2 thai_articles

# ตรวจสอบผลลัพธ์
ls thai_articles/
```

### ดาวน์โหลดและสกัด Wikipedia ภาษาอังกฤษ
```bash
# ดาวน์โหลด dump file (ไฟล์ใหญ่มาก!)
curl -L -o enwiki-latest.xml.bz2 https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# สกัดข้อความ
python extract_wiki.py enwiki-latest.xml.bz2 english_articles
```

## การปรับแต่ง

หากต้องการปรับแต่งการทำงาน สามารถแก้ไขไฟล์ `extract_wiki.py`:

- `articles_per_file`: จำนวนบทความต่อไฟล์ (ค่าเริ่มต้น: 100)
- `min_text_length`: ความยาวข้อความขั้นต่ำของบทความ (ค่าเริ่มต้น: 100 ตัวอักษร)

## หมายเหตุ

- การสกัดข้อมูลอาจใช้เวลานานขึ้นอยู่กับขนาดของ dump file
- สำหรับ Wikipedia ภาษาไทย จะใช้เวลาประมาณ 10-30 นาที
- สำหรับ Wikipedia ภาษาอังกฤษ อาจใช้เวลาหลายชั่วโมง
- ผลลัพธ์จะมีขนาดเล็กกว่า dump file ต้นฉบับมาก

## การแก้ไขปัญหา

### ปัญหา: `wget: command not found`
**วิธีแก้**: ใช้ `curl` แทน `wget`

### ปัญหา: WikiExtractor ไม่ทำงานกับ Python 3.13
**วิธีแก้**: ใช้ไฟล์ `extract_wiki.py` ที่มาพร้อมกับโปรเจกต์นี้แทน

### ปัญหา: หน่วยความจำไม่เพียงพอ
**วิธีแก้**: แบ่งการประมวลผลเป็นส่วนเล็กๆ หรือลดจำนวน `articles_per_file`

ถ้าคุณอยาก **dump ข้อมูลจากวิกิพีเดียภาษาไทย** (`thwiki`) จาก [https://dumps.wikimedia.org](https://dumps.wikimedia.org) นี่คือขั้นตอนที่ชัดเจนและใช้งานได้จริง:

---

## ✅ 1. ไปยังโฟลเดอร์ของ Thai Wikipedia

🔗 [https://dumps.wikimedia.org/thwiki/](https://dumps.wikimedia.org/thwiki/)

ที่หน้านี้คุณจะเห็นรายการของวันที่ที่ Wikimedia ทำการสร้าง dump เช่น:

```
20250601/
20250520/
...
```

---

## ✅ 2. เข้าไปในโฟลเดอร์ที่คุณต้องการ (เช่น `20250601/`)

🔗 ตัวอย่าง: [https://dumps.wikimedia.org/thwiki/20250601/](https://dumps.wikimedia.org/thwiki/20250601/)

คุณจะเห็นไฟล์หลายประเภท เช่น:

| ไฟล์                                                       | ความหมาย                                                                                           |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `thwiki-20250601-pages-articles-multistream.xml.bz2`       | ✅ **ใช้บ่อยที่สุด**: มีบทความทั้งหมด (เฉพาะเนื้อหา ไม่รวมหน้าพูดคุย) พร้อมข้อความในรูปแบบ wikitext |
| `thwiki-20250601-pages-articles-multist# 🌟 WikiDumper - เครื่องมือสกัดข้อมูลวิกิพีเดีย

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**เครื่องมือที่ทันสมัยและใช้งานง่ายสำหรับสกัดข้อความจาก Wikipedia dump files**

รองรับ **Python 3.13** และออกแบบมาเพื่อการใช้งานกับภาษาไทยโดยเฉพาะ

</div>

---

## 🎯 ภาพรวม

WikiDumper เป็นเครื่องมือที่พัฒนาขึ้นเพื่อแก้ปัญหาความเข้ากันได้ของ WikiExtractor เดิมกับ Python รุ่นใหม่ 
โดยมุ่งเน้นให้การสกัดข้อมูลจาก Wikipedia เป็นเรื่องง่ายและรวดเร็ว

### ✨ จุดเด่น

<table>
<tr>
<td align="center">🚀</td>
<td><strong>ทันสมัย</strong><br>รองรับ Python 3.13 และใช้ mwxml library ที่มีประสิทธิภาพ</td>
</tr>
<tr>
<td align="center">🇹🇭</td>
<td><strong>ภาษาไทย</strong><br>ปรับแต่งการทำความสะอาดข้อความสำหรับภาษาไทยโดยเฉพาะ</td>
</tr>
<tr>
<td align="center">🧹</td>
<td><strong>สะอาด</strong><br>ลบ wiki markup, template, references อัตโนมัติ</td>
</tr>
<tr>
<td align="center">📂</td>
<td><strong>จัดระเบียบ</strong><br>แบ่งผลลัพธ์เป็นไฟล์ย่อยขนาดเหมาะสม</td>
</tr>
<tr>
<td align="center">⚡</td>
<td><strong>รวดเร็ว</strong><br>ประมวลผลเร็วกว่า WikiExtractor เดิม</td>
</tr>
</table>

---

## 🔧 การเตรียมพร้อม

### ความต้องการของระบบ

```yaml
✅ Python: 3.7+ (แนะนำ 3.9 หรือใหม่กว่า)
✅ Git: สำหรับ clone repository  
✅ อินเทอร์เน็ต: สำหรับดาวน์โหลด dump files
✅ พื้นที่ว่าง: 2-5 GB (ขึ้นอยู่กับภาษา)
```

### 🛠️ การติดตั้ง

<details>
<summary><strong>📋 ขั้นตอนการติดตั้ง</strong></summary>

**1. Clone โปรเจกต์**
```bash
git clone https://github.com/your-username/wikiDDumper.git
cd wikiDDumper
```

**2. ติดตั้ง Dependencies**
```bash
pip install mwxml
```

**3. ตรวจสอบการติดตั้ง**
```bash
python extract_wiki.py --help
```

</details>

---

## 🚀 วิธีใช้งาน

### ขั้นตอนที่ 1: ดาวน์โหลด Wikipedia Dump

<div align="center">

| 🇹🇭 ภาษาไทย | 🌍 ภาษาอื่น ๆ |
|-------------|-------------|
| **~440 MB** | **ดูตารางด้านล่าง** |

</div>

#### 🇹🇭 สำหรับ Wikipedia ภาษาไทย

```bash
curl -L -o thwiki-20250601-pages-articles-multistream.xml.bz2 \
https://dumps.wikimedia.org/thwiki/20250601/thwiki-20250601-pages-articles-multistream.xml.bz2
```

#### 🌍 สำหรับภาษาอื่น

<details>
<summary><strong>ตัวอย่างภาษายอดนิยม (คลิกเพื่อดู)</strong></summary>

```bash
# 🇺🇸 อังกฤษ (ขนาดใหญ่ ~20GB)
curl -L -o enwiki-latest-pages-articles.xml.bz2 \
https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# 🇯🇵 ญี่ปุ่น (~3GB)
curl -L -o jawiki-latest-pages-articles.xml.bz2 \
https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2

# 🇰🇷 เกาหลี (~1.5GB)
curl -L -o kowiki-latest-pages-articles.xml.bz2 \
https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2

# 🇨🇳 จีน (~4GB)
curl -L -o zhwiki-latest-pages-articles.xml.bz2 \
https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2
```

💡 **เคล็ดลับ:** เยี่ยมชม [dumps.wikimedia.org](https://dumps.wikimedia.org/) เพื่อเลือกภาษาอื่น ๆ

</details>

### ขั้นตอนที่ 2: สกัดข้อความ

#### 🎯 คำสั่งพื้นฐาน

```bash
python extract_wiki.py <input_file> <output_directory>
```

#### 💡 ตัวอย่างการใช้

```bash
# สำหรับภาษาไทย
python extract_wiki.py thwiki-20250601-pages-articles-multistream.xml.bz2 thai_output

# สำหรับภาษาอังกฤษ
python extract_wiki.py enwiki-latest-pages-articles.xml.bz2 english_output
```

---

## ⚙️ การกำหนดค่า

### 📊 พารามิเตอร์หลัก

| พารามิเตอร์ | คำอธิบาย | ตัวอย่าง |
|-------------|----------|---------|
| `input_file` | ไฟล์ Wikipedia dump | `thwiki-latest.xml.bz2` |
| `output_directory` | โฟลเดอร์เก็บผลลัพธ์ | `thai_articles` |

### 🔧 การปรับแต่งขั้นสูง

<details>
<summary><strong>แก้ไขในไฟล์ extract_wiki.py</strong></summary>

```python
# กำหนดจำนวนบทความต่อไฟล์
articles_per_file = 100

# กำหนดความยาวข้อความขั้นต่ำ (ตัวอักษร)
min_text_length = 100

# เปิด/ปิดการแสดงความคืบหน้า
show_progress = True
```

</details>

---

## 📊 ผลลัพธ์

### 📁 โครงสร้างไฟล์ที่ได้

```
output_directory/
├── wiki_00          # บทความที่ 1-100
├── wiki_01          # บทความที่ 101-200  
├── wiki_02          # บทความที่ 201-300
├── wiki_03          # บทความที่ 301-400
└── ...              # และต่อไป
```

### 🎨 รูปแบบเนื้อหา

```xml
<doc id="12345" title="กรุงเทพมหานคร">
กรุงเทพมหานคร เป็นเมืองหลวงและเมืองที่ใหญ่ที่สุดของประเทศไทย
เป็นศูนย์กลางการปกครอง เศรษฐกิจ การศึกษา การคมนาคม
และเป็นที่ตั้งของสถาบันราชวงศ์...
</doc>

<doc id="67890" title="ประเทศไทย">
ประเทศไทย เป็นประเทศในภูมิภาคเอเชียตะวันออกเฉียงใต้
มีพื้นที่ทั้งหมด 513,120 ตารางกิโลเมตร
มีประชากรกว่า 69 ล้านคน...
</doc>
```

---

## 💡 ตัวอย่างการใช้งาน

<details>
<summary><strong>🇹🇭 ตัวอย่างที่ 1: Wikipedia ภาษาไทย</strong></summary>

```bash
# ดาวน์โหลด
curl -L -o thwiki-latest.xml.bz2 \
https://dumps.wikimedia.org/thwiki/latest/thwiki-latest-pages-articles.xml.bz2

# สกัดข้อความ
python extract_wiki.py thwiki-latest.xml.bz2 thai_articles

# ดูผลลัพธ์
ls thai_articles/
head thai_articles/wiki_00
```

</details>

<details>
<summary><strong>🤖 ตัวอย่างที่ 2: การใช้กับ AI/ML</strong></summary>

```bash
# สกัดข้อมูลภาษาไทยสำหรับฝึกโมเดล
python extract_wiki.py thwiki-latest.xml.bz2 training_data

# รวมไฟล์ทั้งหมดเป็นไฟล์เดียว
cat training_data/wiki_* > thai_corpus.txt

# นับจำนวนบรรทัด
wc -l thai_corpus.txt
```

</details>

<details>
<summary><strong>🔍 ตัวอย่างที่ 3: การวิเคราะห์ข้อมูล</strong></summary>

```bash
# หาบทความที่มีคำว่า "กรุงเทพ"
grep -l "กรุงเทพ" thai_articles/wiki_*

# นับจำนวนบทความทั้งหมด
grep -c "<doc" thai_articles/wiki_* | awk -F: '{sum += $2} END {print sum}'
```

</details>

---

## 📈 ประสิทธิภาพ

<div align="center">

| 🌍 ภาษา | 📦 ขนาด Dump | ⏱️ เวลาประมวลผล | 📄 ผลลัพธ์ | 📊 บทความ |
|---------|-------------|----------------|----------|----------|
| 🇹🇭 ไทย | ~440 MB | 10-30 นาที | ~50 MB | ~130,000 |
| 🇯🇵 ญี่ปุ่น | ~3 GB | 1-2 ชั่วโมง | ~400 MB | ~1.3M |
| 🇺🇸 อังกฤษ | ~20 GB | 5-8 ชั่วโมง | ~3 GB | ~6.5M |

</div>

---

## 🔧 การแก้ไขปัญหา

### ❌ ปัญหาที่พบบ่อย

<details>
<summary><strong>wget: command not found</strong></summary>

**วิธีแก้:** ใช้ `curl` แทน `wget`

```bash
# ❌ แทนที่
wget URL

# ✅ ใช้
curl -L -O URL
```

</details>

<details>
<summary><strong>WikiExtractor ไม่ทำงานกับ Python 3.13</strong></summary>

**วิธีแก้:** ใช้ `extract_wiki.py` ที่มาพร้อมโปรเจกต์

✅ เครื่องมือนี้ได้รับการออกแบบมาเพื่อแก้ปัญหานี้โดยเฉพาะ

</details>

<details>
<summary><strong>หน่วยความจำไม่เพียงพอ</strong></summary>

**วิธีแก้:** ลดค่า `articles_per_file` ในไฟล์ `extract_wiki.py`

```python
# เปลี่ยนจาก 100 เป็น 50 หรือน้อยกว่า
articles_per_file = 50
```

</details>

### 💡 เคล็ดลับการใช้งาน

> **ประหยัดเวลา:** ใช้ไฟล์ `latest` แทนวันที่เฉพาะ  
> **ประหยัดพื้นที่:** ลบไฟล์ dump หลังสกัดเสร็จ  
> **เพิ่มความเร็ว:** ใช้ SSD สำหรับเก็บไฟล์ชั่วคราว  

---

## 📚 แหล่งข้อมูลเพิ่มเติม

<div align="center">

| 🔗 ลิงก์ | 📄 คำอธิบาย |
|---------|-------------|
| [Wikipedia Dumps](https://dumps.wikimedia.org/) | แหล่ง dump files ทั้งหมด |
| [mwxml Library](https://github.com/mediawiki-utilities/python-mwxml) | Library หลักที่ใช้ |
| [Original WikiExtractor](https://github.com/attardi/wikiextractor) | เครื่องมือต้นฉบับ |

</div>

---

## 🆘 การสนับสนุน

<div align="center">

หากพบปัญหาหรือต้องการความช่วยเหลือ:

[![Issues](https://img.shields.io/badge/🐛_สร้าง_Issue-GitHub-red?style=for-the-badge)](https://github.com/your-username/wikiDDumper/issues)
[![Documentation](https://img.shields.io/badge/📖_อ่าน_Docs-Wiki-blue?style=for-the-badge)](https://github.com/your-username/wikiDDumper/wiki)
[![Discussions](https://img.shields.io/badge/💬_ถาม_Community-Discussions-green?style=for-the-badge)](https://github.com/your-username/wikiDDumper/discussions)

</div>

---

## 📝 หมายเหตุสำคัญ

- ⏱️ **เวลาประมวลผล** ขึ้นอยู่กับขนาดไฟล์และสเปคเครื่อง
- 💾 **ผลลัพธ์** มีขนาดเล็กกว่า dump file ต้นฉบับมาก (ประมาณ 10-15%)
- 🔄 **Dump files ใหม่** จะออกทุกเดือน
- 🌐 **รองรับทุกภาษา** ใน Wikipedia (300+ ภาษา)
- 🆓 **ใช้งานฟรี** ภายใต้ MIT License

---

<div align="center">

**🌟 สร้างด้วย ❤️ สำหรับชุมชน Open Source 🌟**

[![Star](https://img.shields.io/badge/⭐_Star_this_project-GitHub-yellow?style=for-the-badge)](https://github.com/your-username/wikiDDumper)
[![Fork](https://img.shields.io/badge/🍴_Fork-GitHub-blue?style=for-the-badge)](https://github.com/your-username/wikiDDumper/fork)
[![Contribute](https://img.shields.io/badge/📝_Contribute-GitHub-green?style=for-the-badge)](https://github.com/your-username/wikiDDumper/blob/main/CONTRIBUTING.md)

---

### 🙏 ขอบคุณ

ขอบคุณทุกคนที่ใช้งานและสนับสนุนโปรเจกต์นี้  
หวังว่าเครื่องมือนี้จะเป็นประโยชน์สำหรับงานวิจัยและพัฒนาของคุณ

</div>
ream-index.txt.bz2` | ดัชนีบทความในไฟล์ข้างบน                                                                            |
| `thwiki-20250601-pages-meta-history.xml.bz2`               | 🧠 ขนาดใหญ่มาก: ประวัติการแก้ไขทุกเวอร์ชันของทุกหน้า                                               |
| `thwiki-20250601-categorylinks.sql.gz`, `pagelinks.sql.gz` | ข้อมูลโครงสร้าง เช่น ลิงก์, หมวดหมู่                                                               |

---

## ✅ 3. ดาวน์โหลดไฟล์ที่ต้องการ

### วิธีดาวน์โหลดผ่าน `wget`

```bash
wget https://dumps.wikimedia.org/thwiki/20250601/thwiki-20250601-pages-articles-multistream.xml.bz2
```

---

## ✅ 4. แปลงให้ใช้งานง่าย (เช่น Extract เนื้อหา)

ใช้ [WikiExtractor](https://github.com/attardi/wikiextractor):

```bash
git clone https://github.com/attardi/wikiextractor.git
cd wikiextractor
python3 WikiExtractor.py -o output_dir thwiki-20250601-pages-articles-multistream.xml.bz2
```

จะได้ไฟล์ `.txt` ที่แบ่งเป็นบทความแบบ plaintext พร้อมใช้ต่อ เช่น:

* สำหรับฝึกโมเดล
* ทำ NLP
* วิเคราะห์ข้อความ


## การสนับสนุน

หากพบปัญหาหรือต้องการความช่วยเหลือ กรุณาสร้าง issue ใน GitHub repository นี้


---

## อ้างอิง

- [Wikipedia Dumps](https://dumps.wikimedia.org/)
- [mwxml Library](https://github.com/mediawiki-utilities/python-mwxml)
- [Original WikiExtractor](https://github.com/attardi/wikiextractor)
