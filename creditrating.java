import java.io.*;
import java.util.*;

public class creditrating {
    // args = creditrating [train] [test] [minleaf]
    public static void main(String[] args) throws IOException {
        //===============testing==============
        if (args.length==0)  args = new String[]{"creditrating", "previous.txt", "new.txt", "20"};
        
        Borrower brr = new Borrower();
        ArrayList <Borrower> trainData = new ArrayList<Borrower>();
        ArrayList <Borrower> testData = new ArrayList<Borrower>();
        
        try {
            int minleaf = Integer.parseInt(args[3]);
//            creditrating cr = new creditrating();
            // System.out.printf("[Train]: %s [Test] %s [minleaf]: %s \n", args[1], args[2], minleaf);
            readFile(args[1],trainData);
            readFile(args[2],testData);
            DTL dtl = new DTL();
            dtl.learning(trainData,minleaf);   // info gain
            // List<Double> training = dtl.classify(trainData);
            /* System.out.print("ID3 with full tree on training\t");
            System.out.println(dtl.computeAccuracy(training, trainData));
            List<Double> predictions = dtl.classify(testData);
            System.out.print("ID3 with full tree on test\t")
            System.out.println(dtl.computeAccuracy(predictions, testData));
            */
        } catch (Exception e){
            System.out.println("Error: javac inputfile fail");
        }
    }
    
    public static void readFile (String path, ArrayList <Borrower> list) throws IOException {
        Borrower aBorrower  = new Borrower();
        try{
            BufferedReader in = new BufferedReader(new FileReader(path));
            int id = 1;
//            System.out.println(in.readLine());
            String str;
            while ((str = in.readLine()) != null && id <= 100) {
                String [] arr = str.split("[\\s]+");
                //  or (String s : arr)  System.out.print("&&"+s);
                aBorrower = new Borrower(arr, id);
                aBorrower.printBorrower();
                list.add(aBorrower);
                id++;
            }
            in.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

class Borrower {
    static int featureNumber = 5; // fts can only be 0,1,2,3,4
    int id;     // every borrower has an uniqe Id;
    double WC_TA;
    double RE_TA;
    double EBIT_TA;
    double MVE_BVTD;
    double S_TA;
    String rating;
    double [] features = new double [] {WC_TA,RE_TA,EBIT_TA,MVE_BVTD,S_TA};
    
    // constructor
    public Borrower() { }
    
    public Borrower(String[] str, int id) {
        // for (String s:str)  System.out.println("$$"+s+"  ");
        this.id = id;
        this.WC_TA = Double.parseDouble(str[1]);
        this.RE_TA = Double.parseDouble(str[2]);
        this.EBIT_TA = Double.parseDouble(str[3]);
        this.MVE_BVTD = Double.parseDouble(str[4]);
        this.S_TA = Double.parseDouble(str[5]);
        if (str.length > 6)  {
            this.rating = str[6];
        }
    }
    
    public void printBorrower(){
        System.out.printf("%12.5f %12.5f %12.5f %12.5f %12.5f %12S",WC_TA ,RE_TA, EBIT_TA, MVE_BVTD, S_TA,rating);
        System.out.printf("  ID %-4d \n",id);
    }
}

class DTL {
    Node aNode = new Node();
    String ftName;
    double bestSplitval;
    int bestAttrNum;
    ArrayList<Borrower> leftData = new ArrayList<Borrower>();
    ArrayList<Borrower> rightData = new ArrayList<Borrower>();
    
    // learning(trainData,minleaf);
    public Node learning (ArrayList<Borrower> data, int minleaf) {
        boolean pure = false;
        boolean identical = false;
        
        // decided if data are pure
        for (int i = 0; i < data.size(); i++) {
            for (int j = i+1; j < data.size(); j++) {
                // all borrowers have a same rating
                if (!data.get(i).rating.equals(data.get(j).rating)) {
                    pure = false;
                    break;
                } else {
                    pure = true;
                }
            }
        }
        // decided if data are identical
        for (int i = 0; i < data.size(); i++) {
            for (int j = i+1; j < data.size(); j++) {
                // borrowers have identical features
                if (!Arrays.equals(data.get(i).features, data.get(j).features)) {
                    break;
                }
                else {
                    identical = true;
                }
            }
        }
        
        // Base Case: Stop when N<= minleaf or they are pure or identical
        if (data.size() <= minleaf || pure || identical) {
            //unique node)
            if (pure) {
                aNode.label = data.get(0).rating;
            } else {
                aNode.label = "unknown";
            }
            return aNode;
        }
    
        //[attr,splitval]  choose-split(data)
        bestAttrNum = Integer.parseInt(chooseSplit(data).split("_",2)[0]);
        bestSplitval = Double.parseDouble(chooseSplit(data).split("_",2)[1]);
        splitData(data,bestAttrNum,bestSplitval);
        
        // recursively call learning function
        aNode.left = learning(leftData,minleaf);
        aNode.right = learning(rightData,minleaf);
        return aNode;
    }
    
    public String chooseSplit (ArrayList<Borrower> data) {
        double bestGain = 0;
        double aGain;
        double aSplitval = 0.0;
    
        //for each attr
//        ArrayList<double[]> attr;
        for (int attrNum = 0; attrNum < 5 ; attrNum++) {
            //sort the array X1[attr] X2 [attr]
            sortAttr(data, attrNum);
            // calculate splitval & info gain
            for (int i = 0; i < data.size(); i++) {
                aSplitval = 0.5 * (data.get(i).features[attrNum] + data.get(i).features[attrNum]);
                aGain = infoGain(data, attrNum, aSplitval);
                // attr.add(data.get(i).features);
                if (aGain > bestGain) {
                    bestGain = aGain;
                    bestAttrNum = attrNum;
                    bestSplitval = aSplitval;
                }
            }
        }
        System.out.println("best gain: " + bestGain);
        System.out.println("best Attr: " + bestAttrNum);
        System.out.println("best splitval: " + bestSplitval);
        //split method;
        return Integer.toString(bestAttrNum) + "_" + Double. toString(bestSplitval);
    }
    
    //Impurity function using entropy.
    public double infoGain (ArrayList<Borrower> data, int attrNum, double aSplitval) {
        splitData(data,attrNum,aSplitval);
        double [] count = classify(data,attrNum,aSplitval);
        double [] countLeft = classify(leftData,attrNum,aSplitval);
        double [] countRight = classify(rightData,attrNum,aSplitval);
        double info = Entropy(data, count ,attrNum);
        double infoLeft = Entropy(leftData,countLeft,attrNum);
        double infoRight = Entropy(rightData,countRight,attrNum);
        double info_att =
                (double)(leftData.size()/data.size())*infoLeft + (double)(rightData.size()/data.size())*infoRight;
        double aGain = info - info_att;
        return aGain;
    }
    
    public void splitData(ArrayList<Borrower> data, int attrNum, double aSplitval) {
//        ArrayList<Borrower> leftData = new ArrayList<Borrower>();
//        ArrayList<Borrower> rightData = new ArrayList<Borrower>();
        for (int i = 0; i < data.size(); i++) {
            if (data.get(i).features[attrNum] <= aSplitval) {
                leftData.add(data.get(i));
            }
            else {
                rightData.add(data.get(i));
            }
        }
    }
    
    public double[] classify(ArrayList<Borrower> data, int attrNum, double aSplitval) {
        double [] count = {0,0,0,0,0,0,0,0,0};
        for (Borrower brr:data) {
            switch (brr.rating){
                case "AAA":
                    count[0]++;
                    break;
                case "AA":
                    count[1]++;
                    break;
                case "A":
                    count[2]++;
                    break;
                case "BBB":
                    count[3]++;
                    break;
                case "BB":
                    count[4]++;
                    break;
                case "B":
                    count[5]++;
                    break;
                case "CCC":
                    count[6]++;
                    break;
                case "CC":
                    count[7]++;
                    break;
                case "C":
                    count[8]++;
                    break;
                default:
                    break;
            }
        }
        return count;
    }
    
    
    public double Entropy(ArrayList<Borrower> data, double[] count, int attrNum){
        double [] possibility = new double[count.length];
    
        for (int i = 0; i < count.length; i++) {
            possibility[i] = count [i]/data.size();
        }
    
        double Ent = 0;
        for (int i = 0; i < count.length; i++) {
            if (count[i]>0){
                Ent += -possibility[i] * Math.log(possibility[i]);
            }
        }
        return Ent;
    }
    
    public void sortAttr(ArrayList<Borrower> data, int attrNum){
        Borrower tmp = new Borrower();
        for (int i =0; i<data.size()-1; i++) {
            for ( int j=i+1; j<data.size();j++) {
                // compare average
                if (data.get(i).features[attrNum]<data.get(j).features[attrNum]){
                    tmp = data.get(i);
                    data.set(i,data.get(j));
                    data.set(j,tmp);
                }
            }
        }
    }
}

class Node {
    String label;
    String attr;
    double splitval;
    Node left;
    Node right;
    List<Borrower> borrowers;
    int testFts;
    double predictedRating = -1;
    
    Node(){}
    Node(String label){
        this.label = label;
    }
}
