"""
Extraction and summarization prompts for graph ingestion.

Ported from lightrag/prompt.py — only ingestion-related prompts.
"""

from __future__ import annotations

TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"
GRAPH_FIELD_SEP = "<SEP>"

DEFAULT_ENTITY_TYPES = [
    "Person", "Creature", "Organization", "Location", "Event",
    "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject",
]
DEFAULT_LANGUAGE = "English"

ENTITY_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

ENTITY_EXTRACTION_USER_PROMPT = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

ENTITY_CONTINUE_EXTRACTION_PROMPT = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

ENTITY_EXTRACTION_EXAMPLES = [
    """<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

<Output>
entity<|#|>Alex<|#|>person<|#|>Alex is a character who experiences frustration and is observant of the dynamics among other characters.
entity<|#|>Taylor<|#|>person<|#|>Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.
entity<|#|>Jordan<|#|>person<|#|>Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.
entity<|#|>Cruz<|#|>person<|#|>Cruz is associated with a vision of control and order, influencing the dynamics among other characters.
entity<|#|>The Device<|#|>equipment<|#|>The Device is central to the story, with potential game-changing implications, and is revered by Taylor.
relation<|#|>Alex<|#|>Taylor<|#|>power dynamics, observation<|#|>Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device.
relation<|#|>Alex<|#|>Jordan<|#|>shared goals, rebellion<|#|>Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.)
relation<|#|>Taylor<|#|>Jordan<|#|>conflict resolution, mutual respect<|#|>Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.
relation<|#|>Jordan<|#|>Cruz<|#|>ideological conflict, rebellion<|#|>Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.
relation<|#|>Taylor<|#|>The Device<|#|>reverence, technological significance<|#|>Taylor shows reverence towards the device, indicating its importance and potential impact.
<|COMPLETE|>

""",
    """<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, nexon technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

<Output>
entity<|#|>Global Tech Index<|#|>category<|#|>The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today.
entity<|#|>Nexon Technologies<|#|>organization<|#|>Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings.
entity<|#|>Omega Energy<|#|>organization<|#|>Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices.
entity<|#|>Gold Futures<|#|>product<|#|>Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets.
entity<|#|>Crude Oil<|#|>product<|#|>Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand.
entity<|#|>Market Selloff<|#|>category<|#|>Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations.
entity<|#|>Federal Reserve Policy Announcement<|#|>category<|#|>The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability.
entity<|#|>3.4% Decline<|#|>category<|#|>The Global Tech Index experienced a 3.4% decline in midday trading.
relation<|#|>Global Tech Index<|#|>Market Selloff<|#|>market performance, investor sentiment<|#|>The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns.
relation<|#|>Nexon Technologies<|#|>Global Tech Index<|#|>company impact, index movement<|#|>Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index.
relation<|#|>Gold Futures<|#|>Market Selloff<|#|>market reaction, safe-haven investment<|#|>Gold prices rose as investors sought safe-haven assets during the market selloff.
relation<|#|>Federal Reserve Policy Announcement<|#|>Market Selloff<|#|>interest rate impact, financial regulation<|#|>Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff.
<|COMPLETE|>

""",
    """<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

<Output>
entity<|#|>World Athletics Championship<|#|>event<|#|>The World Athletics Championship is a global sports competition featuring top athletes in track and field.
entity<|#|>Tokyo<|#|>location<|#|>Tokyo is the host city of the World Athletics Championship.
entity<|#|>Noah Carter<|#|>person<|#|>Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship.
entity<|#|>100m Sprint Record<|#|>category<|#|>The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter.
entity<|#|>Carbon-Fiber Spikes<|#|>equipment<|#|>Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction.
entity<|#|>World Athletics Federation<|#|>organization<|#|>The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations.
relation<|#|>World Athletics Championship<|#|>Tokyo<|#|>event location, international competition<|#|>The World Athletics Championship is being hosted in Tokyo.
relation<|#|>Noah Carter<|#|>100m Sprint Record<|#|>athlete achievement, record-breaking<|#|>Noah Carter set a new 100m sprint record at the championship.
relation<|#|>Noah Carter<|#|>Carbon-Fiber Spikes<|#|>athletic equipment, performance boost<|#|>Noah Carter used carbon-fiber spikes to enhance performance during the race.
relation<|#|>Noah Carter<|#|>World Athletics Championship<|#|>athlete participation, competition<|#|>Noah Carter is competing at the World Athletics Championship.
<|COMPLETE|>

""",
]

SUMMARIZE_DESCRIPTIONS_PROMPT = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""
