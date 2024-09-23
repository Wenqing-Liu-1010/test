import torch
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity

# 加载模型
angle = AnglE.from_pretrained(
    'NousResearch/Llama-2-7b-hf',
    pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
    pooling_strategy='last',
    is_llm=True,
    torch_dtype=torch.float16
).cuda()

# 示例数据
data = [
    {
        "context_text": "A company launches a revolutionary product.",
        "supporter_text": "The product's unique features attract a quite large customer base.",
        "defeater_text": "Competitors quickly release similar products, reducing the company’s advantage.",
        "neutral_text": "People frequently share their happy usage experience on social media."
    },
    {
        "context_text": "A new restaurant opens in town.",
        "supporter_text": "The restaurant offers a diverse menu that attracts food enthusiasts.",
        "defeater_text": "Critics say the food quality has declined in recent reviews.",
        "neutral_text": "Local customers have mixed feelings about the service."
    },
    {
        "context_text": "A smartphone brand releases a new model.",
        "supporter_text": "The new model has innovative features that wow consumers.",
        "defeater_text": "Some users complain about the price increase.",
        "neutral_text": "Online reviews vary in opinion."
    },
    {
        "context_text": "A tech conference is being held next month.",
        "supporter_text": "Industry leaders will share insights into the future of technology.",
        "defeater_text": "Some people believe that conferences are a waste of time.",
        "neutral_text": "Attendees look forward to networking opportunities."
    },
    {
        "context_text": "A non-profit organization is launching a new initiative.",
        "supporter_text": "The initiative aims to help underprivileged children.",
        "defeater_text": "Some question the effectiveness of such initiatives.",
        "neutral_text": "The community is invited to participate."
    },
    {
        "context_text": "A new movie is released this weekend.",
        "supporter_text": "Critics are raving about its stunning visuals and engaging storyline.",
        "defeater_text": "Some audiences find the plot predictable.",
        "neutral_text": "Fans are excited to see their favorite actors."
    },
    {
        "context_text": "A local park is undergoing renovations.",
        "supporter_text": "The improvements will provide better facilities for families.",
        "defeater_text": "Some believe the park is fine as it is.",
        "neutral_text": "Visitors are curious about the changes."
    },
    {
        "context_text": "A new fitness app launches today.",
        "supporter_text": "Users praise its user-friendly interface and helpful features.",
        "defeater_text": "Some question its effectiveness compared to traditional methods.",
        "neutral_text": "Many are eager to try it out."
    },
    {
        "context_text": "A new educational program is introduced.",
        "supporter_text": "The program is designed to enhance students' learning experiences.",
        "defeater_text": "Critics argue that it lacks practical application.",
        "neutral_text": "Teachers are hopeful about its impact."
    },
    {
        "context_text": "A fashion brand releases its latest collection.",
        "supporter_text": "Fashion enthusiasts are excited about the fresh styles.",
        "defeater_text": "Some criticize the prices as being too high.",
        "neutral_text": "Reviews are mixed regarding the new line."
    }
]

# 编码文本并计算相似度
results = []
for entry in data:
    context = entry['context_text']
    supporter = entry['supporter_text']
    defeater = entry['defeater_text']
    neutral = entry['neutral_text']

    texts = [supporter, defeater, neutral]
    doc_vecs = angle.encode([{'text': text} for text in texts], prompt=Prompts.A)

    # 计算余弦相似度
    similarity_neutral_supporter = cosine_similarity(doc_vecs[2], doc_vecs[0])
    similarity_neutral_defeater = cosine_similarity(doc_vecs[2], doc_vecs[1])
    similarity_supporter_defeater = cosine_similarity(doc_vecs[0], doc_vecs[1])

    results.append({
        "context": context,
        "neutral_supporter_similarity": similarity_neutral_supporter.item(),
        "neutral_defeater_similarity": similarity_neutral_defeater.item(),
        "supporter_defeater_similarity": similarity_supporter_defeater.item(),
    })

# 打印结果
for result in results:
    print(result)
