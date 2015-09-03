package com.imedpower.segment;

import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.List;


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
            List<Term> parse;

            File writefile = new File("out.txt");
            writefile.createNewFile();
            BufferedWriter out = new BufferedWriter(new FileWriter(writefile));

            while (line != null) {
                String[] oldWeibo = line.split("\t");
                parse = NlpAnalysis.parse(oldWeibo[oldWeibo.length - 1]);
                List<String> words = new ArrayList<>();
                for (int i = 0; i < parse.size(); ++i) {
                    Term item = parse.get(i);
                    words.add(item.getRealName());
                    if (item.getRealName() != item.getName()) {
                        System.out.println("name: " + item.getName());
                        System.out.println("realname: " + item.getRealName());
                        System.out.println("\n");
                    }
                }
                oldWeibo[oldWeibo.length - 1] = words.toString();
                List<String> newWeibo = java.util.Arrays.asList(oldWeibo);
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
}
