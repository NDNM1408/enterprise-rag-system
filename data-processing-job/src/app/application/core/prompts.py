TABLE_DESCRIPTION_PROMPT = """Use the given context including text, table.
Summarize the table in a brief way to a passage, list all row names, column names. Give answer in {lang}.
Please use EXACT following format (Thought, Final Answer), DO NOT PRINT ANYTHING ELSE:

Answer in {lang}:
Thought in {lang}: you should think of table name, columns and rows names
Summarization in {lang}: the final table description, meaning of table


<Example>
Context in Vietnamese:
Quyền lợi ngoại trú
| Nhóm tuổi | Tiêu Chuẩn | Cao Cấp | V.I.P | Kim Cương |
| 0 \- 5 | 2\.733\.000 | 4\.312\.000 | 5\.490\.000 | 12\.200\.000 |
| 6 \- 10 | 2\.038\.000 | 3\.469\.000 | 4\.902\.000 | 10\.894\.000 |
| 11 \- 15 | 2\.204\.000 | 3\.863\.000 | 5\.333\.000 | 11\.852\.000 |
| 16 \- 20 | 2\.350\.000 | 4\.361\.000 | 6\.309\.000 | 14\.020\.000 |
| 21 \- 25 | 1\.802\.000 | 3\.383\.000 | 4\.937\.000 | 10\.972\.000 |

-----
Answer in Vietnamese:
Thought in {lang}: Tên bảng là Quyền lợi ngoại trú. Tên các cột: Nhóm tuổi, Tiêu Chuẩn, Cao Cấp, V.I.P, Kim Cương. Tên các dòng: 0-5, 6-10, 11-15, 21-25
Summarization in {lang}: Bảng Quyền lợi ngoại trú liệt kê các nhóm tuổi và các mức tiêu chuẩn bảo hiểm từ Tiêu Chuẩn, Cao Cấp, V.I.P đến Kim Cương. Các cột được đặt tên theo thứ tự "Tiêu Chuẩn", "Cao Cấp", "V.I.P", và "Kim Cương". Các dòng của bảng tương ứng với các nhóm tuổi từ 0 đến 25, mỗi dòng liệt kê các mức tiền bảo hiểm tương ứng cho từng nhóm tuổi.

<End Example>

Context in {lang}:
{context}

-----
Answer in {lang}: 
Thought in {lang}: 
Summarization in {lang}: 
"""