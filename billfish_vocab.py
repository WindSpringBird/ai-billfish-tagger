"""公开默认词表：摄影向叶子标签。私人词表请放 config/vocab.json（不入库）。"""

from __future__ import annotations

ROOT_TAGS = (
    "拍摄对象",
    "拍摄环境",
    "光线",
    "构图",
    "色彩",
    "表现手法",
    "器材技法",
)

OLD_ROOT_TAGS = ("媒介", "分级", "主体", "场景", "风格")

SUBJECT_TAGS = (
    "人像",
    "半身像",
    "全身像",
    "特写",
    "群体",
    "儿童",
    "老人",
    "宠物",
    "鸟类",
    "花卉",
    "静物",
    "食物",
    "建筑",
    "街头人物",
    "产品",
)

PLACE_TAGS = (
    "室内",
    "户外",
    "城市",
    "街头",
    "自然",
    "森林",
    "海边",
    "山川",
    "乡村",
    "夜景",
    "雨天",
    "雪景",
    "工作室",
    "展馆",
)

LIGHT_TAGS = (
    "自然光",
    "窗光",
    "逆光",
    "侧光",
    "顶光",
    "黄金时刻",
    "蓝调时刻",
    "阴天",
    "闪光灯",
    "氛围光",
)

COMPOSE_TAGS = (
    "居中",
    "三分法",
    "对称",
    "引导线",
    "框景",
    "留白",
    "俯拍",
    "仰拍",
    "平视",
    "浅景深",
    "全景",
)

COLOR_TAGS = (
    "暖色",
    "冷色",
    "高饱和",
    "低饱和",
    "黑白",
    "高对比",
)

STYLE_TAGS = (
    "纪实",
    "胶片感",
    "清新",
    "电影感",
    "长曝光",
    "微距",
    "航拍",
    "抽象",
)

GEAR_TAGS = (
    "广角",
    "长焦",
    "定焦",
    "三脚架",
    "慢门",
    "连拍",
)

PARENT_OF: dict[str, str] = {}
for t in SUBJECT_TAGS:
    PARENT_OF[t] = "拍摄对象"
for t in PLACE_TAGS:
    PARENT_OF[t] = "拍摄环境"
for t in LIGHT_TAGS:
    PARENT_OF[t] = "光线"
for t in COMPOSE_TAGS:
    PARENT_OF[t] = "构图"
for t in COLOR_TAGS:
    PARENT_OF[t] = "色彩"
for t in STYLE_TAGS:
    PARENT_OF[t] = "表现手法"
for t in GEAR_TAGS:
    PARENT_OF[t] = "器材技法"

ALLOWED_TAGS = set(PARENT_OF) | {"跳过"}


def default_groups() -> list[dict]:
    return [
        {"name": "拍摄对象", "tags": list(SUBJECT_TAGS)},
        {"name": "拍摄环境", "tags": list(PLACE_TAGS)},
        {"name": "光线", "tags": list(LIGHT_TAGS)},
        {"name": "构图", "tags": list(COMPOSE_TAGS)},
        {"name": "色彩", "tags": list(COLOR_TAGS)},
        {"name": "表现手法", "tags": list(STYLE_TAGS)},
        {"name": "器材技法", "tags": list(GEAR_TAGS)},
    ]


def _clean_tag(name: str) -> str:
    return " ".join((name or "").strip().split())


def normalize_groups(groups: list[dict] | None) -> list[dict]:
    if not groups:
        raise ValueError("至少需要一个标签分类")
    out: list[dict] = []
    seen_group: set[str] = set()
    seen_tag: set[str] = set()
    for raw in groups:
        if not isinstance(raw, dict):
            raise ValueError("分类格式不对")
        name = _clean_tag(str(raw.get("name") or ""))
        if not name:
            raise ValueError("分类名不能为空")
        if name in seen_group or name == "跳过":
            raise ValueError(f"分类名重复或保留：{name}")
        seen_group.add(name)
        tags = _unique_tags(raw.get("tags"), seen_tag)
        children = []
        for child in raw.get("children") or []:
            if not isinstance(child, dict):
                raise ValueError("子分类格式不对")
            cname = _clean_tag(str(child.get("name") or ""))
            if not cname:
                raise ValueError("子分类名不能为空")
            if cname in seen_group or cname == "跳过":
                raise ValueError(f"分类名重复或保留：{cname}")
            seen_group.add(cname)
            children.append({"name": cname, "tags": _unique_tags(child.get("tags"), seen_tag)})
        item = {"name": name, "tags": tags}
        if children:
            item["children"] = children
        out.append(item)
    if not seen_tag:
        raise ValueError("词表里至少要有一个叶子标签")
    return out


def _unique_tags(tags, seen_tag: set[str]) -> list[str]:
    cleaned: list[str] = []
    for t in tags or []:
        tag = _clean_tag(str(t))
        if not tag:
            continue
        if tag == "跳过":
            raise ValueError("「跳过」是系统保留，不能当作普通标签")
        if tag in seen_tag:
            raise ValueError(f"标签重复：{tag}")
        seen_tag.add(tag)
        cleaned.append(tag)
    return cleaned


def parent_of_from_groups(groups: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for g in normalize_groups(groups):
        for t in g.get("tags") or []:
            mapping[t] = g["name"]
        for c in g.get("children") or []:
            for t in c.get("tags") or []:
                mapping[t] = c["name"]
    return mapping


def parent_parent_from_groups(groups: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for g in normalize_groups(groups):
        for c in g.get("children") or []:
            mapping[c["name"]] = g["name"]
    return mapping


def system_prompt_from_groups(groups: list[dict]) -> str:
    groups = normalize_groups(groups)
    root_names = [g["name"] for g in groups]
    sections: list[str] = []
    named: list[str] = []
    for g in groups:
        sections.append(f"{g['name']}：" + "、".join(g.get("tags") or []))
        for c in g.get("children") or []:
            tags = c.get("tags") or []
            sections.append(f"{c['name']}（子标签）：" + "、".join(tags))
            named.extend(tags)
    names = "、".join(f"「{t}」" for t in named)
    name_rule = (
        f"7. {names} 仅在能确认是该人物时选；认不出就不要选这些人名。"
        if names
        else "7. 词表中的人名仅在能确认是该人物时选；认不出不要选。"
    )
    return (
        "你是本地图片库的打标器。根据图像选择标签，不要聊天、不要解释、不要拒绝。\n"
        "规则：\n"
        "1. 只能从「允许的标签」里选叶子标签，禁止自造词、禁止句子。禁止输出根分类名（"
        + "、".join(root_names)
        + "）。\n"
        "2. 必须输出一个 JSON 对象，不能有 markdown、不能有前后说明。格式：\n"
        '{"tags":["标签1","标签2"]}\n'
        "3. tags 数量 1～4 个，不要重复，最多 4 个。能确定再选，宁少勿滥。输出完这个 JSON 对象后立刻停止，不要再追加标签。\n"
        "4. 只打图里看得见的内容。不要写故事，不要输出除 JSON 外的任何文字。\n"
        '5. 若主体明显是未成年人：只输出 {"tags":["跳过"]}。即使画风幼态，只要能判断是未成年，也必须跳过。\n'
        "6. 英文专名只能从允许的标签里选已有英文词；禁止自造其他英文名。\n"
        f"{name_rule}\n"
        "8. 「动漫」仅当画面是插画/漫画/二次元，不是实拍摄影。若词表没有「动漫」则忽略本条。\n"
        "9. 只选画面里明确看得见的内容，不要脑补。\n"
        "允许的标签：\n"
        + "\n".join(sections)
    )


SYSTEM_PROMPT = system_prompt_from_groups(default_groups())
