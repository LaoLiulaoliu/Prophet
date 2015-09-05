package com.imedpower.segment;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


/**
 * Created by Administrator on 2015/8/12.
 * Usage:
 *   https://github.com/NLPchina/ansj_seg
 *   http://nlpchina.github.io/ansj_seg/
 */
public class Main {

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage: java -jar segment-xxx.jar:\n" +
                    "\t-f filename,  file with every string line\n" +
                    "\t-s string, string need to be segment\n");
        } else {
            String[] argument = args[0].split(" ");
            if (args[0].equals("-s")) {
                List<Term> parse = NlpAnalysis.parse(args[1]);
                System.out.println(parse);
            } else if (args[0].equals("-f")) {
                participleFile(args[1]);
            }
        }
    }

    private static void participleFile(String filename) {
        InputStreamReader reader;
        try {
            reader = new InputStreamReader(new FileInputStream(new File(filename)));
            BufferedReader br = new BufferedReader(reader);
            String line = br.readLine();

            File writefile = new File("out.txt");
            writefile.createNewFile();
            BufferedWriter out = new BufferedWriter(new FileWriter(writefile));

            Pattern namePattern = Pattern.compile(atNameRegex(), Pattern.CASE_INSENSITIVE);
            while (line != null) {
                String[] oneLine = line.split("\t");
                String weibo = oneLine[oneLine.length - 1];
                List<String> words = new ArrayList<>();

                //replaceAll http link to lliinnkk
                weibo = weibo.replaceAll(urlRegex(), " lliinnkk ");

                //remember @name content, replace nnaammee back to @name content later
                List<String> atNames = new ArrayList<>();
                Matcher m = namePattern.matcher(weibo);
                while (m.find()) {
                    atNames.add(m.group(1));
                }
                //replaceAll @name to nnaammee
                weibo = weibo.replaceAll(atNameRegex(), " nnaammee ");

                //separate emoji pattern from words, analysis
                words.addAll( emojiPatternNlp(weibo) );

                //replace nnaammee back to @name content
                for (int i = 0, j = 0; i < words.size(); i++) {
                    if (words.get(i).equals("nnaammee")) {
                        words.set(i, atNames.get(j));
                        j++;
                    }
                }

                oneLine[oneLine.length - 1] = words.toString();
                List<String> newWeibo = java.util.Arrays.asList(oneLine);
                String join = StringUtils.join(newWeibo, "\t");
                out.write(join + "\n");
                line = br.readLine();
            }
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String urlRegex() {
        final String regex = "((https|http|ftp|rtsp|mms)?://)"
                + "(([0-9]{1,3}\\\\.){3}[0-9]{1,3}"
                + "|"
                + "([0-9A-Za-z_!~*'()-]+\\.)*"
                + "([0-9A-Za-z][0-9A-Za-z-]{0,61})?[0-9A-Za-z]\\."
                + "[A-Za-z]{2,6})"
                + "(:[0-9]{1,4})?"
                + "((/[0-9A-Za-z_!~*'().;?:@&=+$,%#-]+)+/?|(/?))";
        return regex;
    }

    public static String atNameRegex() {
        final String regex = "(@[a-zA-Z0-9\\u4E00-\\u9FA5]+)( |$|）)";
        return regex;
    }

    public static List<String> parsePhrase(String phrase) {
        if (phrase == null || phrase.equals("")) {
            return null;
        }
        List<Term> parse = NlpAnalysis.parse(phrase);
        List<String> words = new ArrayList<>();
        for (int i = 0; i < parse.size(); ++i) {
            Term item = parse.get(i);
            words.add(item.getRealName());
        }
        return words;
    }

    public static List<String> emojiPatternNlp(String weibo) {
        List<String> words = new ArrayList<>();

        while (true) {
            int begin = weibo.indexOf("[");
            if (begin == -1) {
                if (!weibo.equals("")) {
                    words.addAll(parsePhrase(weibo));
                }
                break;
            } else {
                String beforeEmoji = weibo.substring(0, begin);
                if (!beforeEmoji.equals("")) {
                    words.addAll(parsePhrase(beforeEmoji));
                }
                int end = weibo.indexOf("]");
                words.add(weibo.substring(begin, end + 1));
                weibo = weibo.substring(end + 1);
            }
        }
        return words;
    }
}
