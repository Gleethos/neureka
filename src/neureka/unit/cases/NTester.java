package neureka.unit.cases;

import neureka.frame.NMessageFrame;

public class NTester {

    protected NMessageFrame Console;
    protected NMessageFrame ResultConsole;
    protected String bar = "[|]";
    protected String line = "--------------------------------------------------------------------------------------------";
    private int positiveAssertions = 0;
    private int assertionCount = 0;

    private int success = 0;
    private int tests = 0;

    private String session = "";

    protected NTester(String name) {
        this.Console = new NMessageFrame(name+" - TEST PROCESS");
        this.ResultConsole = new NMessageFrame(name+" - TEST RESULT");
    }

    protected void printSessionStart(String message){
        this.session = "";
        tests++;
        this.positiveAssertions = 0;
        this.assertionCount = 0;
        this.Console.println("");
        this.Console.println(bar+"  "+message);
        this.Console.println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        this.Console.println(bar+line);
        printlnResult("");
        printlnResult(bar+"  "+message);
        printlnResult("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        printlnResult(bar);
    }
    protected int printSessionEnd(){
        success += (positiveAssertions== assertionCount)?1:0;
        this.Console.println(bar+"  "+((positiveAssertions >0)?"test successful!"+" "+ positiveAssertions :"test failed!"+" "+(assertionCount + positiveAssertions))+"/"+ assertionCount);
        this.Console.println("[O][=][=][=][=][=][=][=][=][=][=][=]|> "+success+"/"+tests);
        this.ResultConsole.println(bar+"  "+((positiveAssertions >0)?"test successful!"+" "+ positiveAssertions :"test failed!"+" "+(assertionCount + positiveAssertions))+"/"+ assertionCount);
        this.ResultConsole.println("[O][=][=][=][=][=][=][=][=][=][=][=]|> "+success+"/"+tests);
        return positiveAssertions;
    }

    protected boolean assertEqual(double result, double expected){
        assertionCount++;
        if(result==expected) {
            println(bar+"  [result]:("+result+") == [expected]:("+expected+") -> test successful.");
            positiveAssertions = (positiveAssertions <0)? positiveAssertions : positiveAssertions +1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  [result]:("+result+") =|= [expected]:("+expected+") -> test failed!");
            positiveAssertions = (positiveAssertions <0)? positiveAssertions -1:-1;
            println(bar+line);
            return false;
        }
    }

    protected boolean assertContains(String name, String result, String expected){
        assertionCount++;
        if(result.contains(expected)) {
            println(bar+"  ["+name+"]:("+result+") "+((result.length()>22)?"\n"+bar+"    contains   -> test successful!"+"\n"+bar+" ":"contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test successful!"));
            positiveAssertions = (positiveAssertions <0)? positiveAssertions : positiveAssertions +1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  ["+name+"]:("+result+") "
                +((result.length()>22)
                    ?"\n"+bar+"    not contains   -> test failed!"+"\n"+bar+" "
                    :"not contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            positiveAssertions = (positiveAssertions <0)? positiveAssertions -1:-1;
            println(bar+line);
            return false;
        }
    }

    protected boolean assertEqual(String name, String result, String expected){
        assertionCount++;
        if(result.equals(expected)) {
            println(bar+"  ["+name+"]:("+result+") "+((result.length()>22)
                    ?"\n"+bar+"    ==   -> test successful!"+"\n"+bar+" ":"==")+" [expected]:("+expected+")"+((result.length()>22)?""
                    :" -> test successful!"));
            positiveAssertions = (positiveAssertions <0)? positiveAssertions : positiveAssertions +1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  ["+name+"]:("+result+") "
                    +((result.length()>22)
                    ?"\n"+bar+"    =|=   -> test failed!"+"\n"+bar+" "
                    :"=|=")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            positiveAssertions = (positiveAssertions <0)? positiveAssertions -1:-1;
            println(bar+line);
            return false;
        }
    }

    protected boolean assertEqual(String result, String expected){
        assertionCount++;
        if(result.equals(expected)) {
            println(bar+"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+bar+"    ==  "+"\n"+bar+" "
                    :"==")+" [expected]:("+expected+") -> test successful.");
            positiveAssertions = (positiveAssertions <0)? positiveAssertions : positiveAssertions +1;
            println(bar+line);
            return true;
        } else {
            println(bar+"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+bar+"    =|=  "+"\n"+bar+" "
                    :"=|=")+" [expected]:("+expected+") -> test failed!");
            positiveAssertions = (positiveAssertions <0)? positiveAssertions -1:-1;
            println(bar+line);
            return false;
        }
    }

    protected String stringified(int[] a){
        String result = "";
        for(int ai : a) {
            result += ai+", ";
        }
        return result;
    }
    protected String stringified(short[] a){
        String result = "";
        for(short ai : a) {
            result += ai+", ";
        }
        return result;
    }
    public String stringified(double[] a){
        String result = "";
        for(double ai : a) {
            result += ai+", ";
        }
        return result;
    }
    protected String stringified(double[][] a){
        String result = "";
        for(double[] ai : a) {
            result+="(";
            for(double aii : ai){
                result += aii+", ";
            }
            result+="), ";
        }
        return result;
    }
    protected String stringified(float[] a){
        String result = "";
        for(float ai : a) {
            result += ai+", ";
        }
        return result;
    }

    protected void print(String message){
        session+=message;
        Console.print(message);
    }
    public void println(String message){
        session+=message+"\n";
        Console.println(message);
    }
    protected void printResult(String message){
        ResultConsole.print(message);
    }
    protected void printlnResult(String message){
        ResultConsole.println(message);
    }

}
