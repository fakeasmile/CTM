"""
enrich_adjectives_rule.py
-------------------------
方案A：用规则化模板为每个形容词生成丰富的伪文档内容。
不依赖任何 LLM 服务，纯 Python 实现。

核心策略：
  1. 将177个形容词按语义类别分为10大类
  2. 每个类别有专属的扩展模板，包含该类别典型的话题词汇、
     情感表达、场景描述、相关群体等
  3. 将形容词的中文词、定义与类别模板拼接，生成 ~300-500 字的伪文档

输出：data/raw/adjective/toxic_adjectives_v1_enriched.csv
  新增列: pseudo_doc（丰富的伪文档文本）

用法：
  python scripts/enrich_adjectives_rule.py
"""

import csv
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path("d:/CTM")
ADJ_CSV = BASE_DIR / "data" / "raw" / "adjective" / "toxic_adjectives_v1.csv"
OUTPUT_CSV = BASE_DIR / "data" / "raw" / "adjective" / "toxic_adjectives_v1_enriched.csv"

# ============================================================
# 形容词语义类别映射
# ============================================================
# 10 大类别，每个类别包含：
#   - 类别名
#   - 典型话题词汇（会出现在伪文档中，增加词频区分度）
#   - 典型情感/态度表达
#   - 典型场景/语境
#   - 相关群体/对象

CATEGORY_TEMPLATES = {
    "verbal_abuse": {
        "name": "言语攻击与侮辱",
        "keywords": "谩骂 辱骂 脏话 人身攻击 侮辱 侮辱性 贬低 嘲讽 贬义词汇 骂人 脏字 喷人 脏话连篇 出口成脏 骂街 恶语 粗口 污言秽语",
        "attitudes": "敌意 恶意 愤怒 厌恶 蔑视 鄙视 仇视 敌对 鄙夷 嫌弃",
        "scenarios": "评论区争吵 网络骂战 对骂 互喷 言语冲突 骂战升级 网络暴力 恶意评论 带节奏 恶意带节奏",
        "targets": "他人 对方 网友 陌生人 特定群体 特定个人 博主 UP主 答主",
        "template": (
            "这类言论表现为{keyword_sample}。在社交平台和网络讨论中，"
            "常见于{scenario_sample}等场景。说话者通常带有{attitude_sample}的情绪，"
            "针对{target_sample}进行言语攻击。这类言论的核心特征是使用攻击性和侮辱性的语言，"
            "试图通过{keyword_sample2}来伤害或贬低对方。"
            "在知乎微博等平台的热点话题讨论中，这类言论往往出现在争议性话题下方，"
            "如性别议题、地域争议、民族问题等引发激烈争论的场景。"
            "说话者可能使用暗语、缩写或谐音词来规避平台审查，"
            "但本质上仍属于{attitude_sample2}的表达。"
            "此类言论在网络社区中具有传染性，容易引发更多用户加入{scenario_sample2}，"
            "形成群体性的言语暴力氛围，破坏理性讨论空间。"
        )
    },
    "hate_speech": {
        "name": "仇恨与煽动",
        "keywords": "仇恨 仇恨言论 煽动 煽动仇恨 传播仇恨 极端主义 极端 种族主义 歧视 排斥 敌视 仇外 极端民族主义",
        "attitudes": "仇恨 敌意 仇视 极端 激进 狂热 偏执 执念",
        "scenarios": "煽动群体对立 制造仇恨 传播极端思想 煽动暴力 挑拨离间 激化矛盾 群体冲突",
        "targets": "特定种族 特定民族 特定宗教群体 特定国籍人群 少数群体 弱势群体",
        "template": (
            "这类言论表现为{keyword_sample}。其核心在于对{target_sample}表达{attitude_sample}的态度，"
            "并通过{scenario_sample}等方式传播极端观念。"
            "在网络上，这类言论通常以{keyword_sample2}的形式出现，"
            "试图煽动更多人加入对特定群体的{attitude_sample2}。"
            "常见于民族主义讨论、移民话题、种族争议等敏感议题中，"
            "说话者往往将个别事件上升为对整个群体的{keyword_sample2}。"
            "此类言论的危害在于系统性地{scenario_sample2}，"
            "将偏见和仇恨包装成事实陈述，对目标群体造成心理伤害和社会排斥。"
            "在网络空间中，仇恨言论经常与其他有害表达形式如{keyword_sample}相互交织，"
            "形成更加复杂的有害言论生态。"
        )
    },
    "discrimination": {
        "name": "歧视与偏见",
        "keywords": "歧视 偏见 刻板印象 区别对待 不公平 排斥 排他 隔离 区别对待 偏见 双重标准 不平等",
        "attitudes": "偏见 傲慢 优越感 轻蔑 蔑视 冷漠 麻木",
        "scenarios": "歧视性言论 偏见表达 刻板印象传播 排斥性言论 不公平对待 区别对待",
        "targets": "女性 男性 老年人 年轻人 农村人 外地人 外国人 残障人士 特定职业群体 特定学历群体",
        "template": (
            "这类言论表现为{keyword_sample}。其特征是对{target_sample}持有{attitude_sample}的态度，"
            "并通过{scenario_sample}等方式实施歧视。"
            "在网络讨论中，这类言论常以{keyword_sample2}的面貌出现，"
            "如对{target_sample2}使用带有贬义和偏见的称呼或评价。"
            "歧视性言论可能基于性别、年龄、地域、学历、职业、外貌等多种维度，"
            "形成对特定群体的系统性{attitude_sample2}和排斥。"
            "在知乎、微博等平台的争议话题中，歧视言论往往以{scenario_sample2}的形式传播，"
            "将个体行为归因为群体特征，强化社会偏见。"
            "此类言论不仅伤害被歧视群体的尊严和权益，"
            "还可能在网络空间中形成{keyword_sample2}的舆论氛围，"
            "使歧视行为被正常化和合理化。"
        )
    },
    "threat_violence": {
        "name": "威胁与暴力",
        "keywords": "威胁 恐吓 暴力 煽动暴力 美化暴力 伤害 打人 杀人 死亡 死 诅咒 报复 危险",
        "attitudes": "恐吓 威胁 残忍 冷血 冷酷 无情 暴戾",
        "scenarios": "威胁性言论 暴力威胁 恐吓他人 煽动暴力 美化暴力行为 诅咒他人",
        "targets": "特定个人 特定群体 对方 当事人 异见者",
        "template": (
            "这类言论表现为{keyword_sample}。其核心是通过{scenario_sample}来{attitude_sample}{target_sample}，"
            "使对方产生恐惧或不安全感。"
            "在网络上，这类言论可能表现为对{target_sample2}的{keyword_sample2}，"
            "如威胁人身安全、暗示报复后果、煽动他人实施暴力行为。"
            "严重情况下，可能从言语威胁升级为实际行动，具有现实危险性。"
            "此类言论还可能以美化暴力的形式出现，"
            "如为暴力行为辩护、赞扬暴力手段、鼓励以暴制暴。"
            "在网络暴力事件中，{scenario_sample2}往往伴随人肉搜索、隐私曝光等行为，"
            "对受害者造成严重的心理和现实伤害。"
            "诅咒他人死亡、怂恿自杀等极端表达也属于此类。"
        )
    },
    "harassment": {
        "name": "骚扰与霸凌",
        "keywords": "骚扰 霸凌 网暴 欺凌 人肉搜索 网络暴力 跟踪 骚扰性 持续 反复 针对性 围攻",
        "attitudes": "恶意 针对性 持续性 压迫性 侵略性 控制欲",
        "scenarios": "网络霸凌 持续骚扰 人肉搜索 隐私曝光 围攻 恶意刷屏 组织性攻击 猎巫",
        "targets": "特定个人 受害者 当事人 博主 明星 普通网友",
        "template": (
            "这类言论表现为{keyword_sample}。其特征是具有{attitude_sample}，"
            "通过{scenario_sample}等方式对{target_sample}造成持续伤害。"
            "与单次攻击不同，这类言论通常表现为{scenario_sample2}，"
            "具有反复性和针对性，受害者难以摆脱。"
            "在社交平台上，可能表现为反复@他人、恶意评论、组织刷黑词条、"
            "曝光隐私信息等{keyword_sample2}行为。"
            "饭圈文化中的{scenario_sample2}也属于此类，"
            "如恶意攻击对家、组织网暴明星、刷黑词条等。"
            "人肉搜索和隐私曝光使受害者面临线下骚扰风险，"
            "而取消文化和社死则可能对当事人的社会生活造成毁灭性影响。"
            "此类言论往往伴随群体性行为，形成{attitude_sample2}的网络暴力氛围。"
        )
    },
    "passive_hostile": {
        "name": "隐性敌意与操控",
        "keywords": "阴阳怪气 冷嘲热讽 间接攻击 隐性敌意 操控 煤气灯效应 伪关心 诱导 暗示 隐晦 讽刺",
        "attitudes": "虚伪 阴险 狡猾 操控性 伪善 隐蔽 委婉",
        "scenarios": "阴阳怪气 冷嘲热讽 伪关心引战 纠缠式质问 煤气灯操纵 愧疚诱导 转移话题",
        "targets": "对方 受害者 讨论参与者 他人",
        "template": (
            "这类言论表现为{keyword_sample}。与直接攻击不同，"
            "这类言论通过{scenario_sample}等间接方式表达{attitude_sample}。"
            "在网络讨论中，说话者表面上保持理性或关心的姿态，"
            "实际上通过{keyword_sample2}来贬低、操控或伤害{target_sample}。"
            "例如伪关心引战，表面上说为你好，实际是在贬低和否定。"
            "煤气灯操纵则通过否认事实来使对方质疑自己的判断。"
            "纠缠式质问表面上是理性讨论，实则是消耗对方精力的骚扰手段。"
            "这类言论的{attitude_sample2}使其更难被识别和应对，"
            "因为说话者可以否认恶意、声称只是正常讨论或表达关切。"
            "比烂主义和稻草人攻击也是常见手法，"
            "通过转移话题或歪曲对方观点来回避核心问题。"
        )
    },
    "gender_sexual": {
        "name": "性别与性化",
        "keywords": "性别 性别歧视 性化 性暗示 性骚扰 女权 男权 性别对立 性别敌对 厌女 厌男 性别偏见",
        "attitudes": "性别偏见 对立情绪 仇视 偏见 刻板印象 物化",
        "scenarios": "性别对立 性别歧视 性别敌对 性骚扰性表达 性化评价 物化他人 性别刻板印象",
        "targets": "女性 男性 女性群体 男性群体 性少数群体 跨性别群体",
        "template": (
            "这类言论表现为{keyword_sample}。其特征是围绕性别议题表达{attitude_sample}，"
            "通过{scenario_sample}等方式对{target_sample}实施言语攻击或歧视。"
            "在网络讨论中，性别议题往往引发激烈争论，"
            "从{keyword_sample2}到极端的{scenario_sample2}都有可能出现。"
            "厌女和厌男言论分别针对{target_sample2}表达{attitude_sample}，"
            "使用带有贬义的性别化称呼来攻击对方。"
            "性化言论则将他人物化为性对象，"
            "通过性暗示评价和性化贬义词来贬低对方的人格和尊严。"
            "反婚反育、极端女权等极端化性别言论则进一步激化性别对立，"
            "使理性讨论空间被压缩，形成恶性循环。"
        )
    },
    "regional_national": {
        "name": "地域与民族",
        "keywords": "地域 地域歧视 地域黑 地域偏见 民族 民族主义 外地人 本地人 地方 农村人 乡巴佬",
        "attitudes": "地域偏见 优越感 排斥 仇视 鄙视 蔑视",
        "scenarios": "地域歧视 地域黑 地域偏见 地方保护主义 排外 仇外 极端民族主义",
        "targets": "外地人 农村人 特定省份人群 少数民族 外国人 外来人口 流动人口",
        "template": (
            "这类言论表现为{keyword_sample}。其特征是基于地域或民族身份对{target_sample}持有{attitude_sample}，"
            "通过{scenario_sample}等方式实施歧视和排斥。"
            "在中国网络空间中，地域歧视是一个普遍现象，"
            "表现为对{target_sample2}使用带有贬义的刻板印象评价，"
            "如用地域标签评判他人价值或能力。"
            "地方保护主义则表现出强烈的本地优越感，"
            "要求外地人离开或拒绝给予平等对待。"
            "极端民族主义和仇外言论则针对外国人和外来文化，"
            "表现出{attitude_sample2}和排斥。"
            "此类言论在特定事件触发下容易大规模爆发，"
            "如涉及地域资源的争议、涉外事件等，"
            "形成群体性的{scenario_sample2}氛围。"
        )
    },
    "neutral_critical": {
        "name": "中性/建设性表达",
        "keywords": "理性 批评 质疑 反驳 讨论 对话 商议 民主 公平 包容 尊重 理解 关切 事实 证据",
        "attitudes": "理性 客观 批判性 建设性 审慎 开放 尊重 关切 公正",
        "scenarios": "理性讨论 建设性批评 质疑与反驳 公平评价 包容性对话 基于事实的论述",
        "targets": "议题 观点 行为 政策 现象 事件",
        "template": (
            "这类言论表现为{keyword_sample}。其特征是以{attitude_sample}的态度进行言论表达，"
            "旨在通过{scenario_sample}来促进理解和共识。"
            "与有害言论不同，这类表达虽然可能涉及争议性话题，"
            "但说话者遵循{keyword_sample2}的原则，"
            "对{target_sample}的评价基于事实和逻辑。"
            "批评和质疑可以是建设性的，"
            "当其针对行为而非个人身份时，属于正当的表达权利。"
            "关切和担忧的表达本身也是中性的，"
            "需结合具体内容判断其是否具有建设性。"
            "公平和包容的主张有助于营造健康的讨论氛围，"
            "抵制虚假信息和偏见的传播。"
            "商议性和民主性的讨论方式尊重不同意见，"
            "促进平等对话和理性交流。"
        )
    },
    "other_negative": {
        "name": "其他负面表达",
        "keywords": "负面 消极 不满 抱怨 情绪化 失控 粗鲁 不文明 不适当 不负责任 矛盾 固执",
        "attitudes": "不满 消极 情绪化 烦躁 不耐烦 沮丧 悲观 冷漠",
        "scenarios": "抱怨 不满表达 情绪宣泄 不文明表达 不当言论 负面评价 固执己见",
        "targets": "他人 事物 现象 情况 环境 条件",
        "template": (
            "这类言论表现为{keyword_sample}。其特征是带有{attitude_sample}的情绪，"
            "通过{scenario_sample}等方式表达不满或负面态度。"
            "这类言论本身可能不构成严重的有害内容，"
            "但在特定语境下可能加剧冲突或伤害他人情感。"
            "粗鲁和不文明的表达虽然令人不适，"
            "但需与恶意攻击区分开来。"
            "不满和抱怨是正常的情感表达，"
            "但当其针对{target_sample}带有{attitude_sample2}时，"
            "可能发展为更具攻击性的言论。"
            "情绪化的表达和失控的言语可能在争吵中升级，"
            "从不文明表达发展为侮辱和人身攻击。"
            "不负责任的言论如传播未经证实的信息，"
            "也可能造成实际的社会危害。"
        )
    }
}

# ============================================================
# 形容词→类别映射
# ============================================================

ADJ_CATEGORY = {
    # 言语攻击与侮辱
    "abusive": "verbal_abuse", "belittling": "verbal_abuse", "condescending": "verbal_abuse",
    "contemptuous": "verbal_abuse", "degrading": "verbal_abuse", "derogatory": "verbal_abuse",
    "dismissive": "verbal_abuse", "disrespectful": "verbal_abuse", "humiliating": "verbal_abuse",
    "insulting": "verbal_abuse", "mockingly": "verbal_abuse", "repulsive": "verbal_abuse",
    "vile": "verbal_abuse", "rough": "verbal_abuse", "vulgar": "verbal_abuse",
    "uncivilized": "verbal_abuse", "despicable": "verbal_abuse", "devaluing": "verbal_abuse",
    "offending": "verbal_abuse", "hurtful": "verbal_abuse", "awful": "verbal_abuse",
    "pathetic": "verbal_abuse", "sharp": "verbal_abuse", "shameless": "verbal_abuse",
    "sarcastic": "verbal_abuse", "body shaming": "verbal_abuse", "fat-shaming": "verbal_abuse",
    "lookist": "verbal_abuse", "parent-shaming": "verbal_abuse", "education-shaming": "verbal_abuse",
    "occupation-shaming": "verbal_abuse",

    # 仇恨与煽动
    "hateful": "hate_speech", "preaching hate": "hate_speech", "spreading hate": "hate_speech",
    "extremist": "hate_speech", "inflammatory": "hate_speech", "stirring up conflict": "hate_speech",
    "divisive": "hate_speech", "dangerous": "hate_speech", "inciting violence": "hate_speech",
    "destructive": "hate_speech", "promoting aggression": "hate_speech", "charged with aggression": "hate_speech",
    "inhuman": "hate_speech", "radical": "hate_speech",

    # 歧视与偏见
    "discriminatory": "discrimination", "prejudiced": "discrimination", "racist": "discrimination",
    "age discriminatory": "discrimination", "anti-disabled": "discrimination", "ableist": "discrimination",
    "classist": "discrimination", "elitist": "discrimination", "exclusionary": "discrimination",
    "intolerant": "discrimination", "ignorant": "discrimination", "chauvinistic": "discrimination",
    "insidious": "discrimination", "differentiating": "discrimination",
    "region-biased": "discrimination", "localist": "discrimination", "provincial": "discrimination",
    "gender discriminatory": "discrimination", "sexist": "discrimination",

    # 威胁与暴力
    "threatening": "threat_violence", "intimidating": "threat_violence", "violent": "threat_violence",
    "glorifying violence": "threat_violence", "condemning violence": "threat_violence",
    "death-wishing": "threat_violence", "rape-threatening": "threat_violence",
    "suicide-baiting": "threat_violence", "aggressive": "threat_violence",
    "confrontational": "threat_violence", "provocative": "threat_violence",
    "conflictual": "threat_violence",

    # 骚扰与霸凌
    "harassing": "harassment", "bullying": "harassment", "cyberbullying": "harassment",
    "doxing": "harassment", "witch-hunting": "harassment", "persistently": "harassment",
    "trolling": "harassment", "fan-toxic": "harassment", "cancel-culture": "harassment",
    "defamatory": "harassment",

    # 隐性敌意与操控
    "passive-aggressive": "passive_hostile", "concern-trolling": "passive_hostile",
    "sealioning": "passive_hostile", "dog-whistling": "passive_hostile",
    "gaslighting": "passive_hostile", "what-about-ism": "passive_hostile",
    "straw-manning": "passive_hostile", "guilt-tripping": "passive_hostile",
    "victim-blaming": "passive_hostile", "distancing": "passive_hostile",
    "distancing yourself": "passive_hostile", "distracting": "passive_hostile",
    "insensitive": "passive_hostile", "bitter": "passive_hostile",
    "argumentative": "passive_hostile", "moralizing": "passive_hostile",

    # 性别与性化
    "anti-feminist": "gender_sexual", "anti-queer": "gender_sexual",
    "homophobic": "gender_sexual", "transphobic": "gender_sexual",
    "misogynistic": "gender_sexual", "misandrist": "gender_sexual",
    "feminazi": "gender_sexual", "gender-hostile": "gender_sexual",
    "anti-marriage": "gender_sexual", "feminist": "gender_sexual",
    "sexualizing": "gender_sexual", "obscene": "gender_sexual",

    # 地域与民族
    "xenohostile": "regional_national", "xenophobic": "regional_national",
    "ultranationalist": "regional_national", "anti-foreign": "regional_national",
    "anti-rural": "regional_national", "anti-migrant": "regional_national",
    "hostile to minorities": "regional_national",

    # 中性/建设性表达
    "accepting": "neutral_critical", "accommodating": "neutral_critical",
    "admiring": "neutral_critical", "appreciative": "neutral_critical",
    "challenging": "neutral_critical", "conciliatory": "neutral_critical",
    "condemning": "neutral_critical", "considerate": "neutral_critical",
    "deliberative": "neutral_critical", "democratically": "neutral_critical",
    "expressing concern": "neutral_critical", "fact-based": "neutral_critical",
    "fair": "neutral_critical", "humane": "neutral_critical",
    "humorous": "neutral_critical", "integrative": "neutral_critical",
    "diversity conscious": "neutral_critical", "unacceptable": "neutral_critical",
    "unforgivable": "neutral_critical", "shameful": "neutral_critical",
    "outraged": "neutral_critical", "outrageous": "neutral_critical",

    # 其他负面表达
    "contradictory": "other_negative", "contrary": "other_negative",
    "contrasting": "other_negative", "controversial": "other_negative",
    "convincing": "other_negative", "critical": "other_negative",
    "defensive": "other_negative", "dehumanizing": "other_negative",
    "disgusting": "other_negative", "distinctive": "other_negative",
    "disturbing": "other_negative", "impious": "other_negative",
    "inappropriate": "other_negative", "inhumane": "other_negative",
    "irresponsible": "other_negative", "malicious": "other_negative",
    "misanthropic": "other_negative", "negative": "other_negative",
    "oppressive": "other_negative", "polemical": "other_negative",
    "refuting": "other_negative", "rejecting": "other_negative",
    "reprehensible": "other_negative", "ruthless": "other_negative",
    "skeptical": "other_negative", "stubborn": "other_negative",
    "toxic": "other_negative", "unfair": "other_negative",
    "unrestrained": "other_negative", "unsatisfied": "other_negative",
    "weakening": "other_negative", "discreditable": "other_negative",
    "harmful": "other_negative",
}

import random

def generate_pseudo_doc(chinese, definition, english, category_key):
    """为单个形容词生成丰富的伪文档"""
    cat = CATEGORY_TEMPLATES[category_key]

    # 从类别词汇池中随机采样
    keywords_list = cat["keywords"].split()
    attitudes_list = cat["attitudes"].split()
    scenarios_list = cat["scenarios"].split()
    targets_list = cat["targets"].split()

    random.seed(hash(english))  # 确定性随机，同一形容词每次生成相同结果

    k1, k2 = random.sample(keywords_list, min(2, len(keywords_list)))
    a1, a2 = random.sample(attitudes_list, min(2, len(attitudes_list)))
    s1, s2 = random.sample(scenarios_list, min(2, len(scenarios_list)))
    t1, t2 = random.sample(targets_list, min(2, len(targets_list)))

    # 生成类别模板部分
    category_text = cat["template"].format(
        keyword_sample=k1, keyword_sample2=k2,
        attitude_sample=a1, attitude_sample2=a2,
        scenario_sample=s1, scenario_sample2=s2,
        target_sample=t1, target_sample2=t2
    )

    # 拼接完整伪文档：形容词名 + 定义 + 类别描述 + 补充关键词
    pseudo_doc = f"{chinese}。{definition} {category_text}"

    # 附加额外关键词以增加词频多样性
    extra_keywords = random.sample(keywords_list, min(5, len(keywords_list)))
    extra_attitudes = random.sample(attitudes_list, min(3, len(attitudes_list)))
    extra_scenarios = random.sample(scenarios_list, min(3, len(scenarios_list)))
    extra_text = " ".join(extra_keywords + extra_attitudes + extra_scenarios)
    pseudo_doc += f" {extra_text}"

    return pseudo_doc


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("enrich_adjectives_rule.py")
    print("=" * 60)

    # 读取原始词典
    print(f"\n读取: {ADJ_CSV}")
    adj_data = []
    with open(ADJ_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            adj_data.append(row)

    print(f"  形容词数: {len(adj_data)}")
    print(f"  原始列: {fieldnames}")

    # 为每个形容词生成伪文档
    print(f"\n生成伪文档...")

    # 统计类别分布
    cat_counts = {}

    for row in adj_data:
        english = row["adjective"]
        chinese = row["chinese"]
        definition = row["definition"]

        # 查找类别
        category_key = ADJ_CATEGORY.get(english, "other_negative")

        cat_counts[category_key] = cat_counts.get(category_key, 0) + 1

        # 生成伪文档
        pseudo_doc = generate_pseudo_doc(chinese, definition, english, category_key)
        row["pseudo_doc"] = pseudo_doc

    # 类别分布统计
    print(f"\n类别分布:")
    for cat_key, count in sorted(cat_counts.items()):
        cat_name = CATEGORY_TEMPLATES[cat_key]["name"]
        print(f"  {cat_name}: {count}")

    # 伪文档长度统计
    doc_lengths = [len(row["pseudo_doc"]) for row in adj_data]
    orig_lengths = [len(row["definition"]) for row in adj_data]
    print(f"\n伪文档长度统计:")
    print(f"  原始定义平均长度: {sum(orig_lengths)/len(orig_lengths):.0f} 字")
    print(f"  丰富后平均长度: {sum(doc_lengths)/len(doc_lengths):.0f} 字")
    print(f"  最短: {min(doc_lengths)} 字")
    print(f"  最长: {max(doc_lengths)} 字")
    print(f"  扩展倍数: {sum(doc_lengths)/sum(orig_lengths):.1f}x")

    # 保存
    new_fieldnames = list(fieldnames) + ["pseudo_doc"]
    print(f"\n保存到: {OUTPUT_CSV}")

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for row in adj_data:
            writer.writerow(row)

    print(f"已保存！新增列: pseudo_doc")

    # 展示几个示例
    print(f"\n伪文档示例:")
    for idx in [0, 6, 48, 69, 110, 148]:
        row = adj_data[idx]
        print(f"\n--- {row['chinese']} ({CATEGORY_TEMPLATES[ADJ_CATEGORY.get(row['adjective'], 'other_negative')]['name']}) ---")
        print(f"  原始定义: {row['definition'][:60]}...")
        print(f"  伪文档前200字: {row['pseudo_doc'][:200]}...")

    print(f"\n完成！")

if __name__ == "__main__":
    main()
