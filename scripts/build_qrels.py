"""
Queries & Relevance Judgments (Qrels) Generator Script.

1. Defines 200 high-quality, category-specific evaluation queries (10 per category).
2. Generates ground-truth Qrels mapping each query to relevant corpus documents belonging to its target category.
3. Exports data/queries.csv and data/qrels.csv.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import pandas as pd
from core.config import METADATA_CSV, QUERIES_CSV, QRELS_CSV, DATA_DIR

# 10 Curated, realistic topic queries for each of the 20 Newsgroups categories (200 total queries)
CATEGORY_QUERIES = {
    "alt.atheism": [
        "atheism belief existence of god arguments",
        "morality without god religion secular ethics",
        "christianity vs atheism debate evidence",
        "bible contradictions historical accuracy critique",
        "freedom of religion secularism state society",
        "theism agnosticism philosophical reasoning",
        "religious intolerance free speech skepticism",
        "origin of universe creationism evolution debate",
        "meaning of life without divine creator",
        "church state separation constitutional rights"
    ],
    "comp.graphics": [
        "computer graphics 3d rendering software algorithms",
        "ray tracing lighting polygon shading technique",
        "image processing file formats jpeg tiff gif",
        "virtual reality graphics pipeline rendering engine",
        "opengl graphics library 3d animation tools",
        "texture mapping bump mapping surface rendering",
        "color space rgb hsv conversion algorithms",
        "vector graphics rasterization algorithms",
        "gpu acceleration 3d graphics rendering performance",
        "bezier curves splines 3d surface modeling"
    ],
    "comp.os.ms-windows.misc": [
        "windows 3.1 installation setup device drivers",
        "ms-dos system configuration config.sys autoexec.bat",
        "microsoft windows memory management swap file",
        "windows graphics device interface gdi print drivers",
        "windows fonts truetype installation troubleshooting",
        "windows file manager disk partitioning format",
        "microsoft windows dynamic link library dll error",
        "windows 3.1 win32s extension compatibility",
        "windows networking tcp ip protocol setup",
        "windows crash blue screen fatal exception fix"
    ],
    "comp.sys.ibm.pc.hardware": [
        "ibm pc motherboard isa vesa bus controller",
        "ide scsi hard drive controller installation",
        "intel 486 processor CPU clock speed upgrade",
        "pc memory ram simm cache chip installation",
        "vga graphics card monitor resolution setup",
        "sound blaster audio card irq dma conflict",
        "bios setup CMOS battery motherboard jumper",
        "floppy drive 1.44mb controller cable setup",
        "pc power supply unit wattage cooling fan",
        "modem serial port uart 16550 baud rate setup"
    ],
    "comp.sys.mac.hardware": [
        "apple macintosh quadra centris bus architecture",
        "mac scsi hard drive termination cabling setup",
        "macintosh powerbook battery memory upgrade",
        "apple mac monitor resolution adapter display",
        "macintosh system 7 operating system memory management",
        "macintosh keyboard ADB mouse port troubleshooting",
        "apple laserwriter printer postscript networking",
        "macintosh motherboards motorola 68040 processor",
        "mac superdrive floppy disk read write error",
        "macintosh external hard drive scsi ID conflict"
    ],
    "comp.windows.x": [
        "x window system x11 server client architecture",
        "motif window manager toolkit widget library",
        "xlib programming window creation event loop",
        "x11 font server display setup display environment",
        "xterm terminal emulator configuration resources",
        "x11 visual depth color map allocation",
        "open look vs motif window manager design",
        "x11 client authorization xhost xauth security",
        "x11 pixel map pixmap bitblt rendering",
        "x11 keycodes keysyms keyboard event mapping"
    ],
    "misc.forsale": [
        "for sale used computer hardware desktop tower",
        "selling audio stereo speakers receiver amplifier",
        "for sale mountain bike bicycle excellent condition",
        "selling cellular phone wireless pager accessories",
        "for sale video game console games cartridges",
        "selling car vehicle low mileage clean title",
        "for sale camera lens photo equipment tripod",
        "selling musical instrument guitar amplifier case",
        "for sale textbook computer science engineering",
        "selling CD player compact discs music collection"
    ],
    "rec.autos": [
        "car engine oil change transmission fluid maintenance",
        "ford mustang vs chevrolet camaro v8 performance",
        "car brake system pads rotors ABS troubleshooting",
        "tire pressure alignment tread wear suspension",
        "manual transmission clutch replacement shifting gear",
        "car radiator cooling system thermostat water pump",
        "fuel injection intake manifold carburetor tuning",
        "car battery alternator electrical system failure",
        "sports car acceleration handling lap time comparison",
        "car insurance rates safety crash test rating"
    ],
    "rec.motorcycles": [
        "motorcycle engine oil filter spark plug maintenance",
        "honda vs yamaha street bike performance handling",
        "riding helmet leather jacket safety gear armor",
        "motorcycle chain sprocket drive belt adjustment",
        "dirt bike off road trail riding suspension",
        "motorcycle brake fluid lever pads bleeding system",
        "harley davidson cruiser exhaust sound custom bike",
        "motorcycle countersteering cornering technique curve",
        "motorcycle battery charging system stator regulator",
        "touring motorcycle windshield saddlebags long distance"
    ],
    "rec.sport.baseball": [
        "major league baseball world series championship game",
        "pitcher earned run average ERA strikeouts fastball",
        "home run hit RBI batting average grand slam",
        "baseball playoff standings division pennant race",
        "catcher umpire strike zone pitch count calling",
        "baseball team roster trade contract free agent",
        "boston red sox vs new york yankees rivalry",
        "baseball stadium outfield fence field dimensions",
        "designated hitter rule batting lineup strategy",
        "hall of fame baseball player statistics record"
    ],
    "rec.sport.hockey": [
        "national hockey league nhl stanley cup playoffs",
        "hockey goal penalty power play short handed goal",
        "ice hockey goalie save percentage shutout mask",
        "hockey player slap shot wrist shot goal scoring",
        "montreal canadines vs toronto maple leafs nhl",
        "hockey team standings division playoffs finals",
        "hockey referee penalty box slashing high sticking",
        "hockey puck ice rink skates blade sharpening",
        "hockey team roster trade draft picks contract",
        "nhl commissioner player strike lock out season"
    ],
    "sci.crypt": [
        "public key cryptography RSA algorithm encryption",
        "data encryption standard DES key length security",
        "clipper chip escrow encryption privacy government",
        "cryptographic hash function MD5 sha digital signature",
        "pretty good privacy PGP encryption key exchange",
        "cipher text plain text secret key decipher",
        "cryptanalysis frequency attack code breaking cipher",
        "crypto privacy wiretap law communications security",
        "random number generator key generation entropy",
        "zero knowledge proof cryptographic protocol authentication"
    ],
    "sci.electronics": [
        "printed circuit board pcb design soldering components",
        "operational amplifier op-amp circuit resistor capacitor",
        "integrated circuit IC pinout voltage regulator",
        "digital logic gates ttl cmos microcontrollers",
        "oscilloscope signal generator voltage measurement test",
        "power supply transformer rectifier filter capacitor",
        "rf radio frequency transmitter antenna impedance matching",
        "semiconductor diode transistor switching circuit",
        "audio amplifier schematic circuit distortion power",
        "microprocessor bus interface clock crystal oscillator"
    ],
    "sci.med": [
        "medical diagnosis symptoms treatment disease pathology",
        "prescription medication drug side effects dosage",
        "diet nutrition vitamin deficiency health benefits",
        "cancer treatment chemotherapy radiation therapy oncology",
        "heart disease blood pressure cholesterol cardiovascular",
        "infectious disease virus bacteria immune system response",
        "neurology brain disease memory loss nerve disorder",
        "medical study clinical trials research medical journal",
        "surgery operation anesthesia surgeon patient recovery",
        "allergy allergic reaction histamine immune response"
    ],
    "sci.space": [
        "NASA space shuttle launch orbit mission payload",
        "satellite orbit gravity payload launch vehicle trajectory",
        "space station hubble space telescope astronomy observation",
        "lunar moon landing mission Apollo program surface",
        "mars rover exploration planet surface space probe",
        "rocket propulsion liquid fuel engine thrust ISP",
        "solar system planets astronomy orbit galaxy universe",
        "commercial space launch vehicle rocket satellite",
        "deep space exploration radio telescope SETI search",
        "zero gravity microgravity space environment astronaut"
    ],
    "soc.religion.christian": [
        "christian doctrine bible scripture new testament gospel",
        "jesus christ resurrection salvation faith grace god",
        "christian church history theology sacraments prayer",
        "old testament biblical prophecy law covenants god",
        "holy spirit trinity father son christian faith",
        "christian morality sin forgiveness redemption bible",
        "catholic vs protestant theology church tradition",
        "christian worship hymn sermon fellowship congregation",
        "heaven hell eternal life judgment christian view",
        "apostolic succession church fathers early christianity"
    ],
    "talk.politics.guns": [
        "second amendment right to bear arms constitutional law",
        "gun control legislation assault weapon ban crime",
        "firearms safety handguns rifles concealed carry permit",
        "national rifle association NRA gun rights lobby",
        "atf federal firearms license background checks",
        "self defense firearm armed citizen crime prevention",
        "gun violence homicide rate statistics firearm law",
        "automatic weapons semi automatic rifle registration",
        "firearm ammunition caliber bullet ballistic trajectory",
        "supreme court gun law constitutional interpretation"
    ],
    "talk.politics.mideast": [
        "middle east peace process israel palestine conflict",
        "arab israeli war territory borders treaty UN",
        "jerusalem west bank gaza strip peace negotiations",
        "jewish homeland zionism palestinian statehood rights",
        "middle east foreign policy international relations US",
        "lebanon syria jordan diplomatic conflict diplomacy",
        "middle east refugee crisis human rights international law",
        "middle east military defense armed forces security",
        "historical roots middle east conflict history treaty",
        "islamic state middle east governance politics society"
    ],
    "talk.politics.misc": [
        "government policy taxation federal budget economic growth",
        "libertarian freedom individual rights state power",
        "presidential election congress political party campaign",
        "freedom of speech civil liberties constitution rights",
        "welfare state social security government entitlement",
        "foreign policy military defense international trade",
        "health care reform government regulations economy",
        "political corruption lobbyist campaign finance reform",
        "federal reserve interest rates inflation economic policy",
        "supreme court constitutional interpretation federal law"
    ],
    "talk.religion.misc": [
        "world religions comparative theology belief systems",
        "religious cults new age philosophy spiritual practice",
        "buddhism hinduism islam christianity comparative faith",
        "religious freedom tolerance blasphemy morality state",
        "theology nature of god divine revelation faith",
        "paganism occult esotericism ancient religion history",
        "mysticism spiritual experience consciousness meditation",
        "morality secular humanism religious ethics comparison",
        "sacred texts scripture interpretation comparative religion",
        "origin of evil suffering divine justice theology"
    ]
}


def build_queries_and_qrels():
    """Build queries.csv and qrels.csv."""
    print("=" * 60)
    print("GENERATING EVALUATION QUERIES & QRELS")
    print("=" * 60)

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata file missing at {METADATA_CSV}. Run scripts/prepare_data.py first.")

    meta_df = pd.read_csv(METADATA_CSV)

    query_records = []
    qrels_records = []

    q_counter = 1
    for cat_name, q_list in CATEGORY_QUERIES.items():
        # Find all document IDs matching this category
        cat_docs = meta_df[meta_df["category_name"] == cat_name]["doc_id"].tolist()
        
        for query_text in q_list:
            query_id = f"Q{q_counter:03d}"
            query_records.append({
                "query_id": query_id,
                "category": cat_name,
                "query": query_text
            })

            # Create binary qrels relevance = 1 for all matching category documents
            for d_id in cat_docs:
                qrels_records.append({
                    "query_id": query_id,
                    "doc_id": d_id,
                    "relevance": 1
                })

            q_counter += 1

    df_queries = pd.DataFrame(query_records)
    df_qrels = pd.DataFrame(qrels_records)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_queries.to_csv(QUERIES_CSV, index=False)
    df_qrels.to_csv(QRELS_CSV, index=False)

    print(f"[+] Saved {len(df_queries)} evaluation queries to: {QUERIES_CSV}")
    print(f"[+] Saved {len(df_qrels)} ground-truth qrels mappings to: {QRELS_CSV}")

    # Validation assertions
    assert len(df_queries) == 200, f"Expected 200 queries, got {len(df_queries)}"
    assert df_queries["query_id"].nunique() == 200, "Query IDs must be unique!"
    assert set(df_qrels["doc_id"]).issubset(set(meta_df["doc_id"])), "Qrels contain unknown document IDs!"
    print("[+] All query & qrels validation checks passed cleanly!")


if __name__ == "__main__":
    build_queries_and_qrels()
