import telebot

botTimeWeb = telebot.TeleBot('7940475138:AAEuIrmByKa0dnC8pyxVblA8VjZj69VroOc')

user_ready_for_questions = {}


@botTimeWeb.message_handler(commands=['start'])
def start_bot(message):
    first_mess = f"<b>{message.from_user.first_name}</b>, привет!\nЯ умею находить ответы на вопросы, опираясь на отправленный вами документ.\n\nПожалуйста, загрузите файл (.pdf)"
    botTimeWeb.send_message(message.chat.id, first_mess, parse_mode='html')
    user_ready_for_questions[message.chat.id] = False


@botTimeWeb.message_handler(content_types=['document'])
def handle_document(message):
    if message.document.mime_type == 'application/pdf':
        botTimeWeb.send_message(message.chat.id, "Файл принят. Начинаю обработку...")

        # Заглушка обработки

        botTimeWeb.send_message(message.chat.id,
                                "Обработка завершена. Теперь я готов отвечать на ваши вопросы по этому документу")

        user_ready_for_questions[message.chat.id] = True
    else:
        botTimeWeb.send_message(message.chat.id, "Файл должен быть формата PDF")


@botTimeWeb.message_handler(content_types=['text'])
def handle_text_message(message):
    print(f"{message.from_user.first_name} написал: {message.text}")

    if user_ready_for_questions.get(message.chat.id, False):

        botTimeWeb.send_message(message.chat.id, "Ответ.") # Заглушка формирования ответа

        # должно быть if else нашелся ответ или нет
    else:
        botTimeWeb.send_message(message.chat.id, "Для начала работы необходимо загрузить файл")

botTimeWeb.infinity_polling()