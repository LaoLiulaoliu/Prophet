#!/bin/bash

data_path="./data"
dst_path="./gen_data"
if ! test -d "$dst_path" ; then
  mkdir -p "$dst_path"
fi
ZH_TEXT="${dst_path}/wiki.zh.text"
ZH_TEXT_JIAN="${dst_path}/wiki.zh.text.jian"
CONV_CONF="zht2zhs.ini"
Bin="env/bin"

Sogou="${data_path}/sougou"
sogou_all_in_one="${dst_path}/sougou_all_in_one"
sogou_text="${dst_path}/sougou_corpus.txt"
sogou_text_jian="${dst_path}/sougou_corpus.txt.jian"
sogou_text_jian_no_tag="${dst_path}/sougou_corpus.txt.jian.notag"


weibo_train="${data_path}/weibo_train_data.txt"
weibo_predict="${data_path}/weibo_predict_data.txt"
weibo_train_text="${dst_path}/weibo_train_data_text.txt"
weibo_predict_text="${dst_path}/weibo_predict_data_text.txt"
weibo_train_text_jian="${dst_path}/weibo_train_data_text.txt.jian"
weibo_predict_text_jian="${dst_path}/weibo_predict_data_text.txt.jian"

if ! test -s "$ZH_TEXT" ; then
	echo "+ process wiki file"
	${Bin}/python utils/process_wiki.py `pwd`/data/zhwiki-latest-pages-articles.xml.bz2 "$ZH_TEXT"
fi

if ! test -s "$ZH_TEXT_JIAN" ; then
	echo "+ convert to jian ti"
	opencc -i "$ZH_TEXT" -o "$ZH_TEXT_JIAN" -c "$CONV_CONF"
fi


if ! test -s "$sogou_all_in_one" ; then
	echo "+ collect sougou all in one"
	cat ${Sogou}/*.txt > "${sogou_all_in_one}"
fi

if ! test -s "$sogou_text" ; then
	echo "+ creating sougou corpus"
	cat "${sogou_all_in_one}" | iconv -f gbk -t utf-8 -c | grep "<content>" > "${sogou_text}"
fi

if ! test -s "$sogou_text_jian" ; then
	echo "+ converting sougou to jian ti"
	opencc -i "${sogou_text}" -o "${sogou_text_jian}" -c "$CONV_CONF"
fi

if ! test -s "$sogou_text_jian_no_tag" ; then
	echo "+ deleting the content tag"
	sed -r 's/<\/?content>//g' "$sogou_text_jian" | grep -v "^$" > "$sogou_text_jian_no_tag"
fi

if ! test -s "$weibo_train_text" ; then
	echo "+ extracting weibo train data"
	${Bin}/python utils/extract_weibo_text.py "${weibo_train}" "${weibo_train_text}"
fi

if ! test -s "$weibo_train_text_jian" ; then
	echo "+ convting weibo train to jian ti"
	opencc -i "${weibo_train_text}" -o "${weibo_train_text_jian}" -c "$CONV_CONF"
fi

if ! test -s "$weibo_predict_text" ; then
	echo "+ extracting weibo predict data"
	${Bin}/python utils/extract_weibo_text.py "${weibo_predict}" "${weibo_predict_text}"
fi

if ! test -s "$weibo_predict_text_jian" ; then
	echo "+ convting weibo predict to jian ti"
	opencc -i "${weibo_predict_text}" -o "${weibo_predict_text_jian}" -c "$CONV_CONF"
fi

wiki_words="${ZH_TEXT_JIAN}.words"
sogou_words="${sogou_text_jian_no_tag}.words"
weibo_train_words="${weibo_train_text_jian}.words"
weibo_predict_words="${weibo_predict_text_jian}.words"

extract_words() {
	from=$1
	to=$2
	java -jar segment/target/segment-1.0-SNAPSHOT.jar -f "${from}"
	if test -s "./out.txt" ; then
		mv "./out.txt" "$to"
	fi
}

if ! test -s "$wiki_words" ; then
	echo "+ extract wiki words"
	extract_words "${ZH_TEXT_JIAN}" "$wiki_words"
fi

if ! test -s "$sogou_words" ; then
	echo "+ extract sogou_words"
	extract_words "${sogou_text_jian_no_tag}" "${sogou_words}"
fi

if ! test -s "$weibo_train_words" ; then
	echo "+ extract weibo train words"
	extract_words "${weibo_train_text_jian}" "${weibo_train_words}"
fi

if ! test -s "$weibo_predict_words" ; then
	echo "+ extract weibo predict words"
	extract_words "${weibo_predict_text_jian}" "${weibo_predict_words}"
fi


dst_model="./gen_model"
if ! test -d "$dst_model" ; then
	mkdir -p "$dst_model"
fi

vec_size="50 100 200 300 400 500"
phrases="0 1 2 3"
window_size="5 7 9"
algo="1 2"
all_files="${wiki_words} ${sogou_words} ${weibo_train_words} ${weibo_predict_words}"
weibo_files="${weibo_train_words} ${weibo_predict_words}"


for vec in `echo "$vec_size" | tr -s ' ' '\n'` ; do
	for ph in `echo "$phrases" | tr -s ' ' '\n'` ; do
		for ws in `echo "$window_size" | tr -s ' ' '\n'` ; do
			for al in `echo "$algo" | tr -s ' ' '\n'` ; do
				output="${dst_model}/vec_state_s${vec}_p${ph}_w${ws}_t${al}"
				echo "+ Generating word vector by s${vec}_p${ph}_w${ws}_t${al}"
				output_all="${output}.all"
				output_weibo="${output}.weibo"
				if ! test -s "$output_all" ; then
					${Bin}/python utils/train_word2vec.py -o "${output_all}" -p $ph -s $vec -w $ws -t $al ${all_files}
				fi
				if ! test -s "$output_weibo" ; then
					${Bin}/python utils/train_word2vec.py -o "${output_weibo}" -p $ph -s $vec -w $ws -t $al ${weibo_files}
				fi
			done
		done
	done
done
