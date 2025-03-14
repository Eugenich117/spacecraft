'''конвертер формата файлов gfc в ascii'''



import pyshtools
# 1. Загрузка коэффициентов гравитационного потенциала (EGM96)
'''исправляет форматы чисел в необходимые'''
def fix_gfc_format(input_file, output_file):
    """
    Преобразует формат чисел с 'd' в 'e' и удаляет строки с заголовком.
    :param input_file: Путь к исходному файлу .gfc.
    :param output_file: Путь к исправленному файлу .gfc.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        is_data = False  # Флаг начала данных
        for line in infile:
            # Ищем начало данных
            if line.strip().startswith("gfc"):
                is_data = True

            if is_data:
                # Заменяем 'd' на 'e' в числах
                fixed_line = line.replace('d', 'e')
                outfile.write(fixed_line)


# Укажите пути к исходному и исправленному файлам
input_file = r"C:\Users\zheny\Downloads\EGM2008\EGM2008.gfc"
output_file = r"C:\Users\zheny\Downloads\EGM2008\EGM2008_fixed.gfc"

fix_gfc_format(input_file, output_file)
print(f"Файл исправлен и сохранен как: {output_file}")