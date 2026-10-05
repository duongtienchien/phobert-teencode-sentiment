import json
import re

raw_pairs = """
ctrai	con trai
khôg	không
bme	bố mẹ
cta	chúng ta
mih	mình
mqh	mối quan hệ
cgai	con gái
nhữg	những
mng	mọi người
svtn	sinh viên tình nguyện
r	rồi
qtam	quan tâm
thươg	thương
qtâm	quan tâm
chug	chung
trườg	trường
thoy	thôi
đki	đăng ký
atsm	ảo tưởng sức mạnh
ạk	ạ
cv	công việc
vch	vãi chưởng
cùg	cùng
pn	bạn
pjt	biết
thjk	thích
ktra	kiểm tra
nek	nè
cgái	con gái
nthe	như thế
chúg	chúng
kái	cái
tìh	tình
phòg	phòng
lòg	lòng
từg	từng
rằg	rằng
sốg	sống
thuj	thôi
thuơng	thương
càg	càng
đky	đăng ký
bằg	bằng
sviên	sinh viên
ák	á
đág	đáng
nvay	như vậy
nhjeu	nhiều
xg	xuống
zồi	rồi
trag	trang
zữ	dữ
atrai	anh trai
kte	kinh tế
độg	động
lmht	liên minh huyền thoại
gắg	gắng
đzai	đẹp trai
thgian	thời gian
đồg	đồng
btrai	bạn trai
nthê	như thế
hìhì	hì hì
vọg	vọng
hihe	hi he
đôg	đông
răg	răng
thườg	thường
tcảm	tình cảm
đứg	đứng
ksao	không sao
dz	đẹp trai
hjxhjx	hix hix
cmày	chúng mày
xuốg	xuống
nkư	như
lquan	liên quan
tiếg	tiếng
xih	xinh
hìh	hình
thàh	thành
ngke	nghe
tnào	thế nào
tưởg	tưởng
ctrinh	chương trình
phog	phong
hôg	không
zìa	về
kũg	cũng
ntnao	như thế nào
trọg	trọng
nthế	như thế
năg	năng
ngđó	người đó
lquen	làm quen
riêg	riêng
ngag	ngang
hêhê	hê hê
bnhiu	bao nhiêu
ngốk	ngốc
kậu	cậu
kqua	kết quả
htrc	hôm trước
địh	định
gđình	gia đình
giốg	giống
csống	cuộc sống
zùi	rồi
bnhiêu	bao nhiêu
cbị	chuẩn bị
kòn	còn
buôg	buông
csong	cuộc sống
chàg	chàng
chăg	chăng
ngàh	ngành
llac	liên lạc
nkưng	nhưng
nắg	nắng
tíh	tính
khoảg	khoảng
thík	thích
ngđo	người đó
ngkhác	người khác
thẳg	thẳng
kảm	cảm
dàh	dành
júp	giúp
lặg	lặng
vđê	vấn đề
bbè	bạn bè
bóg	bóng
dky	đăng ký
dòg	dòng
uốg	uống
tyêu	tình yêu
snvv	sinh nhật vui vẻ
đthoại	điện thoại
qhe	quan hệ
cviec	công việc
tượg	tượng
qà	quà
thjc	thích
nhưq	nhưng
cđời	cuộc đời
bthường	bình thường
đáh	đánh
xloi	xin lỗi
zám	dám
qtrọng	quan trọng
bìh	bình
lzi	làm gì
qhệ	quan hệ
kủa	của
lz	làm gì
đóg	đóng
cka	cha
lgi	làm gì
nvậy	như vậy
qả	quả
đkiện	điều kiện
nèk	nè
tlai	tương lai
bsĩ	bác sĩ
hkì	học kỳ
vde	vấn đề
chta	chúng ta
òy	rồi
ltinh	linh tinh
ngyeu	người yêu
đthoai	điện thoại
snghĩ	suy nghĩ
nặg	nặng
họk	học
dừg	dừng
hphúc	hạnh phúc
hiha	hi ha
wtâm	quan tâm
thíck	thích
chuện	chuyện
lạh	lạnh
ntnày	như thế này
lúk	lúc
ngía	nghía
mớj	mới
hsơ	hồ sơ
ctraj	con trai
nyêu	người yêu
c	chị
kih	kinh
kb	kết bạn
dthương	dễ thương
ctrình	chương trình
mìnk	mình
mjh	mình
ng	người
vc	vợ chồng
uhm	ừm
thỳ	thì
nyc	người yêu cũ
tks	cảm ơn
nàg	nàng
thôii	thôi
đjên	điên
bgái	bạn gái
xink	xinh
hđộng	hành động
đhọc	đại học
mk	mình
bn	bạn
thik	thích
cj	chị
mn	mọi người
nguoi	người
nógn	nóng
hok	không
ko	không
bik	biết
vs	với
cx	cũng
mik	mình
đc	được
cmt	bình luận
ck	chồng
chk	chồng
ngta	người ta
gđ	gia đình
vk	vợ
ctác	công tác
sg	sài gòn
ae	anh em
ah	à
ạh	ạ
rì	gì
ms	mới
vn	việt nam
nhaa	nha
cũg	cũng
đag	đang
ace	anh chị em
àk	à
uh	ừ
cmnr	con mẹ nó rồi
hnay	hôm nay
ukm	ừm
ctr	chương trình
nch	nói chuyện
nta	người ta
ngèo	nghèo
kêh	kênh
ak	à
ad	admin
j	gì
ny	người yêu
dc	được
qc	quảng cáo
baoh	bao giờ
zui	vui
zẻ	vẻ
tym	tim
aye	anh yêu em
eya	em yêu anh
fb	facebook
insta	instagram
z	vậy
thich	thích
đt	điện thoại
trc	trước
chs	chẳng hiểu sao
đhs	đéo hiểu sao
qá	quá
ntn	như thế nào
wá	quá
zậy	vậy
zô	vô
vđ	vãi đái
vchg	vãi chưởng
sml	sấp mặt lồn
xl	xin lỗi
cmn	con mẹ nó
face	facebook
hjhj	hi hi
vv	vui vẻ
ns	nói
iu	yêu
in4	thông tin
kh	không
zạ	vậy
oy	rồi
jo	giờ
troai	trai
wa	quá
hjx	hix
e	em
ji	gì
ce	chị em
lm	làm
đz	đẹp trai
sr	xin lỗi
ib	nhắn tin
hoy	thôi
k	không
vd	ví dụ
a	anh
unf	hủy kết bạn
cty	công ty
kô	không
hqua	hôm qua
xog	xong
uk	ừ
nhoé	nhé
biet	biết
quí	quý
stk	số tài khoản
đươc	được
nghành	ngành
nvqs	nghĩa vụ quân sự
ngừoi	người
trog	trong
tgian	thời gian
biêt	biết
fải	phải
nguời	người
tđn	thế đéo nào
bth	bình thường
tgdd	thế giới di động
khg	không
nhưg	nhưng
thằng	thằng
đuợc	được
ku	cu
thým	thím
onl	trực tuyến
cmnd	chứng minh nhân dân
sđt	số điện thoại
klq	không liên quan
clmm	cái lồn mẹ mày
cmm	con mẹ mày
vcl	vãi cả lồn
vl	vãi lồn
vkl	vãi cả lồn
dm	địt mẹ
đm	địt mẹ
dcm	địt con mẹ
đcm	địt con mẹ
đmm	địt mẹ mày
clgt	cái lồn gì thế
lol	lồn
loz	lồn
lozz	lồn
vcđ	vãi cả đái
đbh	đéo bao giờ
qq	quằn què
wtf	what the fuck
acc	tài khoản
ytb	youtube
sub	đăng ký
bsvv	buổi sáng vui vẻ
ik	đi
my fen	bạn tôi
fen	bạn
thpt	trung học phổ thông
thằg	thằng
zú	vú
đtqg	đội tuyển quốc gia
"""

teencode_map = {}

for line in raw_pairs.strip().split("\n"):
    parts = line.strip().split("\t")
    if len(parts) == 2:
        k = parts[0].strip().lower()
        v = parts[1].strip().lower()
        # Lọc bỏ từ bị kéo dài ký tự để nhường việc cho Regex xử lý
        if not re.search(r"(\w)\1{2,}", k):
            teencode_map[k] = v

# Sắp xếp theo độ dài giảm dần (ưu tiên chuỗi dài thay thế trước)
sorted_dict = dict(sorted(teencode_map.items(), key=lambda x: len(x[0]), reverse=True))

with open("teencode_dict.json", "w", encoding="utf-8") as f:
    json.dump(sorted_dict, f, ensure_ascii=False, indent=2)

print(f"Đã cập nhật hoàn tất teencode_dict.json với {len(sorted_dict)} mục từ!")