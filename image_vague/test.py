from PIL import Image, ImageDraw, ImageFont

def text_wrap(text, font, max_width):
    lines = []
    words = text.split()
    
    while words:
        line_words = []
        while words and draw.textlength(' '.join(line_words + [words[0]]), font=font) <= max_width:
            line_words.append(words.pop(0))
        
        if not line_words:  # 如果当前单词太长无法放入一行，则单独作为一行
            line_words.append(words.pop(0))

        lines.append(' '.join(line_words))
    
    return '\n'.join(lines)

img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(img)
text = "这是一个测试用的例子，用于展示如何让PIL中的文字按照指定的矩形框进行自动换行。"
max_box_size = (400, 200)  # 文字显示的目标矩形框尺寸
position = (50, 50)       # 开始位置坐标(x,y)

# 加载字体文件，并设置初始字号
default_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  
font_size = int(max_box_size[1]/len(text)) or 10
font = ImageFont.truetype(default_font_path, size=font_size)

wrapped_text = text_wrap(text=text, font=font, max_width=max_box_size[0])
draw.multiline_text(position, wrapped_text, fill="black", font=font, align="left")

plt.imshow(img)
plt.axis('off')  # 不显示坐标轴
plt.show()