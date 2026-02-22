import os
import cv2
import numpy as np
import argparse
#import numpy as np

def compare_images(src, dst, region=None, output_dir="diff_output"):
	# src: 比較元ディレクトリ
	# dst: 比較先ディレクトリ
	# region: (x1, y1, x2, y2) のタプル。None の場合は全体比較
	os.makedirs(output_dir, exist_ok=True)

	src_files = sorted(os.listdir(src))

	for filename in src_files:
		src_path = os.path.join(src, filename)
		dst_path = os.path.join(dst, filename)

		if not os.path.exists(dst_path):
			print(f"[SKIP] {filename} は比較先に存在しません")
			continue

		img1 = cv2.imread(src_path)
		img2 = cv2.imread(dst_path)

		if img1 is None or img2 is None:
			print(f"[ERR ] {filename} の読み込みに失敗しました")
			continue

		"""
		# サイズが違う場合は比較不可
		if img1.shape != img2.shape:
			print(f"[DIFF] {filename} は画像サイズが異なります")
			diff_img = draw_full_bbox(img1)
			cv2.imwrite(os.path.join(output_dir, f"diff_{filename}"), diff_img)
			continue
		"""

		# 比較領域の決定（指定がなければ全体）
		if region is None:
			x1, y1, x2, y2 = 0, 0, img1.shape[1], img1.shape[0]
		else:
			x1, y1, x2, y2 = map(int, region.split(","))

		# ROI(Region of Interest) 画像処理の対象となる領域を抽出
		roi1 = img1[y1:y2, x1:x2]
		roi2 = img2[y1:y2, x1:x2]

		# 余白部分をトリミング
		roi1_trim, margin_x, margin_y = trim_white_border(roi1)
		roi2_trim, margin_x, margin_y = trim_white_border(roi2)

		# 余白部分をトリミングすると、比較元と比較先の画像サイズが異なってしまう
		# そうすると、cv2.absdiff()でサイズエラーになる。
		# エラーを回避するため、画像サイズが小さい方に合わせて切り詰める
		h1, w1 = roi1_trim.shape[:2]
		h2, w2 = roi2_trim.shape[:2]

		h = min(h1, h2)
		w = min(w1, w2)

		roi1_trim = roi1_trim[:h, :w]
		roi2_trim = roi2_trim[:h, :w]
		
		# 比較のアリゴリズムの指定
		SlidingWindow = True
		if SlidingWindow:
			# スライディングウィンドウで走査
			# 20x20 の範囲に差分が5個以上ある領域を検出 (★調整)
			boxes = detect_clustered_diff(roi1_trim, roi2_trim, (20, 20), 5)
		else:
			# ピクセル単位で比較
			boxes = pixel_diff(roi1_trim, roi2_trim, x1, y1)

		if len(boxes) == 0:
			# 差分なし。
			print(f"[ OK ] {filename}")
		else:
			# 差分あり。赤枠を描画
			result = draw_diff_boxes(img2, boxes, [margin_x, margin_y])

			out_path = os.path.join(output_dir, f"diff_{filename}")
			cv2.imwrite(out_path, result)
			print(f"[DIFF] {filename} → {out_path} に差分画像を保存しました")


def detect_clustered_diff(roi1, roi2, win_size, pixel_threshold):
	# roi1, roi2: 比較する2つの画像（同サイズ）
	# win_size: (w, h) のスライディングウィンドウサイズ
	# pixel_threshold: 窓内の差分画素数がこの値以上なら差分ありと判定

	# 差分計算
	# 各画素に対して diff(𝑥,𝑦) = |a(𝑥,𝑦)−b(𝑥,𝑦)| を計算する。結果、値が大きいほどdiffが強くなる。
	# イメージとしては「2枚の画像を重ねて、違うところだけ明るく浮かび上がらせた画像を作る」感じ。
	diff = cv2.absdiff(roi1, roi2)

	# diffはRGBの3チャンネル画像で、以下はRGBをグレースケールに変換して、どれくらい違うか」を
	# 0～255で表すようにしている
	# thresholdは「違いが小さいところは無視し、大きいところだけを白く残す」二値化処理
	# 二値化した値が>30なら、その画像を255(白)にする。それ以外は0(黒)にする。
	gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

	h, w = thresh.shape
	win_w, win_h = win_size

	# 差分がまとまっている領域（赤枠の座標）を格納
	diff_boxes = []

	# スライディングウィンドウで走査
	for y in range(0, h - win_h + 1, win_h):
		for x in range(0, w - win_w + 1, win_w):

			window = thresh[y:y+win_h, x:x+win_w]
			count = cv2.countNonZero(window)

			# ★ 窓内に pixel_threshold 以上の差分があれば差分あり
			if count >= pixel_threshold:
				diff_boxes.append((x, y, win_w, win_h))

	return diff_boxes


def pixel_diff(roi1, roi2, x1, y1):
	# 差分計算
	# 各画素に対して diff(𝑥,𝑦) = |a(𝑥,𝑦)−b(𝑥,𝑦)| を計算する。結果、値が大きいほどdiffが強くなる。
	# イメージとしては「2枚の画像を重ねて、違うところだけ明るく浮かび上がらせた画像を作る」感じ。
	diff = cv2.absdiff(roi1, roi2)

	# diffはRGBの3チャンネル画像で、以下はRGBをグレースケールに変換して、どれくらい違うか」を
	# 0～255で表すようにしている
	# thresholdは「違いが小さいところは無視し、大きいところだけを白く残す」二値化処理
	# 二値化した値が>30なら、その画像を255(白)にする。それ以外は0(黒)にする。
	gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# 差分の領域（赤枠の座標）を格納
	diff_boxes = []

	if len(contours) != 0:
		# 差分がある部分に赤枠を描画（元画像に対して）
		for cnt in contours:
			cx, cy, cw, ch = cv2.boundingRect(cnt)
			diff_boxes.append([(x1 + cx), (y1 + cy), cw, ch])

	return diff_boxes

# ---------------------------------------------------------
# 白余白を自動トリミング
# ---------------------------------------------------------
def trim_white_border(img, threshold=250):
	# 白余白を自動でトリミングする
	# threshold: 白とみなす明るさの閾値
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# 白以外のピクセルを抽出
	mask = gray < threshold
	coords = np.column_stack(np.where(mask))

	if coords.size == 0:
		# 全部白ならそのまま返す
		return img

	y_min, x_min = coords.min(axis=0)
	y_max, x_max = coords.max(axis=0)

	trimmed = img[y_min:y_max+1, x_min:x_max+1]
	return trimmed, x_min, y_min


def draw_diff_boxes(base_img, boxes, offset=(0, 0)):
	# base_img: 赤枠を描画する元画像
	# boxes: (x, y, w, h) のリスト
	# offset: ROI が元画像のどこにあるか（x_offset, y_offset）
	ox, oy = offset
	result = base_img.copy()

	for (x, y, w, h) in boxes:
		cv2.rectangle(result, (ox + x, oy + y), (ox + x + w, oy + y + h), (0, 0, 255), 2)

	return result


def draw_full_bbox(img):
	# サイズが違う場合など、全体を赤枠で囲む
	h, w = img.shape[:2]
	result = img.copy()
	cv2.rectangle(result, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("img1", help="compare image 1")
	parser.add_argument("img2", help="compare image 2")
	parser.add_argument("-r", help="compare area. ex) x1,y1,x2,y2")
	args = parser.parse_args()

	compare_images(args.img1, args.img2, args.r)

