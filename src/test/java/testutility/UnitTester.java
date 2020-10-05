package testutility;
import neureka.Tsr;
import org.junit.Assert;

import java.util.List;

public class UnitTester extends Assert
{
    static String BAR = "[|]";
    private static String LINE = "--------------------------------------------------------------------------------------------";
    private int _positive_assertions = 0;
    private int _assertion_count = 0;

    private int _success = 0;
    private int _tests = 0;

    private String _session = "";

    public UnitTester(String name) {
        println("Test-Session: "+name);
    }

    public int testContains(String result, List<String> expected, String description){
        printSessionStart(description);
        println(BAR + "  Tensor: "+result);
        println(BAR + "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        return (printSessionEnd()>0)?1:0;
    }

    public int testContains(String result, String[] expected, String description){

        printSessionStart(description);
        println(BAR + "  Tensor: "+result);
        println(BAR + "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        for(String element : expected){
            this.assertStringContains("result", result, element);
        }
        return (printSessionEnd()>0)?1:0;
    }

    protected void printSessionStart(String message){
        _session = "";
        _tests++;
        _positive_assertions = 0;
        _assertion_count = 0;
        println("");
        println(BAR +"  "+message);
        println("[O][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=][=]=>");
        println(BAR + LINE);
    }
    protected int printSessionEnd(){
        _success += (_positive_assertions == _assertion_count)?1:0;
        println(BAR +"  "+((_positive_assertions >0)?"test successful!"+" "+ _positive_assertions :"test failed!"+" "+(_assertion_count + _positive_assertions))+"/"+ _assertion_count);
        println("[O][=][=][=][=][=][=][=][=][=][=][=]|> "+ _success +"/"+ _tests);
        if(_positive_assertions != _assertion_count) failSession();
        return _positive_assertions;
    }

    protected boolean assertIsEqual(double result, double expected){
        _assertion_count++;
        if(result==expected) {
            println(BAR +"  [result]:("+result+") == [expected]:("+expected+") -> test successful.");
            _positive_assertions = (_positive_assertions <0)? _positive_assertions : _positive_assertions +1;
            println(BAR + LINE);
            return true;
        } else {
            println(BAR +"  [result]:("+result+") =|= [expected]:("+expected+") -> test failed!");
            _positive_assertions = (_positive_assertions <0)? _positive_assertions -1:-1;
            println(BAR + LINE);
            return false;
        }
    }

    protected boolean assertStringContains(String name, String result, String expected){
        _assertion_count++;
        if(result.contains(expected)) {
            println(BAR +"  ["+name+"]:("+result+") "+((result.length()>22)?"\n"+ BAR +"    contains   -> test successful!"+"\n"+ BAR +" ":"contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test successful!"));
            _positive_assertions = (_positive_assertions <0)? _positive_assertions : _positive_assertions +1;
            println(BAR + LINE);
            assertEquals(true, true);
            return true;
        } else {
            println(BAR +"  ["+name+"]:("+result+") "
                +((result.length()>22)
                    ?"\n"+ BAR +"    not contains   -> test failed!"+"\n"+ BAR +" "
                    :"not contains")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            _positive_assertions = (_positive_assertions <0)? _positive_assertions -1:-1;
            println(BAR + LINE);
            failSession();
            return false;
        }
    }

    protected boolean assertIsEqual(String name, String result, String expected){
        _assertion_count++;
        if(result.equals(expected)) {
            println(BAR +"  ["+name+"]:("+result+") "+((result.length()>22)
                    ?"\n"+ BAR +"    ==   -> test successful!"+"\n"+ BAR +" ":"==")+" [expected]:("+expected+")"+((result.length()>22)?""
                    :" -> test successful!"));
            _positive_assertions = (_positive_assertions <0)? _positive_assertions : _positive_assertions +1;
            println(BAR + LINE);
            assertEquals(true, true);
            return true;
        } else {
            println(BAR +"  ["+name+"]:("+result+") "
                    +((result.length()>22)
                    ?"\n"+ BAR +"    =|=   -> test failed!"+"\n"+ BAR +" "
                    :"=|=")+" [expected]:("+expected+")"+((result.length()>22)?"":" -> test failed!"));
            _positive_assertions = (_positive_assertions <0)? _positive_assertions -1:-1;
            println(BAR + LINE);
            //failSession();
            return false;
        }
    }

    protected boolean assertIsEqual(String result, String expected){
        _assertion_count++;
        if(result.equals(expected)) {
            println(BAR +"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+ BAR +"    ==  "+"\n"+ BAR +" "
                    :"==")+" [expected]:("+expected+") -> test successful.");
            _positive_assertions = (_positive_assertions <0)? _positive_assertions : _positive_assertions +1;
            println(BAR + LINE);
            assertTrue(true);
            return true;
        } else {
            println(BAR +"  [result]:("+result+") "+((result.length()>22)
                    ?"\n"+ BAR +"    =|=  "+"\n"+ BAR +" "
                    :"=|=")+" [expected]:("+expected+") -> test failed!");
            _positive_assertions = (_positive_assertions <0)? _positive_assertions -1:-1;
            println(BAR + LINE);
            //failSession();
            return false;
        }
    }

    protected void failSession(){
        fail(_session);
    }

    protected String stringified(int[] a){
        StringBuilder result = new StringBuilder();
        for(int ai : a) {
            result.append(ai).append(", ");
        }
        return result.toString();
    }
    protected String stringified(short[] a){
        StringBuilder result = new StringBuilder();
        for(short ai : a) {
            result.append(ai).append(", ");
        }
        return result.toString();
    }
    public String stringified(double[] a){
        if(a==null){
            return "null";
        }
        StringBuilder result = new StringBuilder();
        for(double ai : a) {
            result.append(Tsr.Utility.Stringify.formatFP(ai)).append(", ");
        }
        return result.toString();
    }
    protected String stringified(double[][] a){
        StringBuilder result = new StringBuilder();
        for(double[] ai : a) {
            result.append("(");
            for(double aii : ai){
                result.append(aii).append(", ");
            }
            result.append("), ");
        }
        return result.toString();
    }
    protected String stringified(float[] a){
        StringBuilder result = new StringBuilder();
        for(float ai : a) {
            result.append(ai).append(", ");
        }
        return result.toString();
    }

    protected void print(String message){
        _session +=message;
    }
    protected void println(String message){
        _session +=message+"\n";
    }

}
