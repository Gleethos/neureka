package neureka;

import java.util.Map;
import java.util.HashMap;
import java.util.function.Consumer;
import java.lang.reflect.Method;
import java.lang.reflect.Field;
import java.util.Vector;

public class NSubstrate {

    private static Map<String, Consumer<String>> Actions;
    private static Map<String, NRealm> Realms;
    private static Map<String, Class> Classes;
    private static NRealm Realm;

    String url = "[Realm]:(null)> ";

    static {
        Actions = new HashMap<String, Consumer<String>>();
        Realms = new HashMap<String, NRealm>();
        Classes = new HashMap<String, Class>();
        initialize();
    }
    private static void initialize(){
        Field f;
        try {
            f = ClassLoader.class.getDeclaredField("classes");
            f.setAccessible(true);
            ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
            Vector<Class> classes =  (Vector<Class>) f.get(classLoader);
            for(Class cls : classes){
                //java.net.URL location = cls.getResource('/' + cls.getName().replace('.',
                //        '/') + ".class");
                //System.out.println("<p>"+location +"<p/> ... "+cls.getName());
                Classes.put(cls.getName(), cls);
                String[] expl = cls.getName().split("\\.");
                Classes.put(expl[expl.length-1], cls);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void setRealm(String name){
        Realm = Realms.get(name);
        url = "[Realm]:(null)> ";
    }

    public static void printRealms(String search){
        System.out.println(search);
    }

    public synchronized static void execute(String command){

        String[] parts = command.split("\\.");
        String[][] broken = new String[parts.length][];
        for(int i=0; i<parts.length; i++){
            broken[i] = parts[i].split("[\\(||\\)]");//This needs to be a 'unpack' function!
            if(broken[i].length>1){
                String[] params = broken[i][1].split(",");
                String[] component = new String[1+params.length];
                component[0] = broken[i][0];
                for(int ii=1; ii<component.length; ii++){
                    component[ii] = params[ii-1];
                }
                broken[i] = component;
            }
        }
        Class[][] ParamClasses = new Class[broken.length][];
        Object[][] Params = new Object[broken.length][];
        for(int i=0; i<broken.length; i++){
            ParamClasses[i] = null;
            if(broken[i].length>1){
                ParamClasses[i] = new Class[broken[i].length-1];
                Params[i] = new Object[broken[i].length-1];

                String regexDecimal = "^-?\\d*\\.\\d+$";
                String regexInteger = "^-?\\d+$";
                String regexDouble = regexDecimal + "|" + regexInteger;

                for(int ii=1; ii<broken[i].length; ii++){
                    ParamClasses[i][ii-1] =
                            (broken[i][ii].contains("\"")||broken[i][ii].contains("'"))
                                ? String.class
                                : ParamClasses[i][ii-1];
                    ParamClasses[i][ii-1] =
                            (broken[i][ii].matches(regexInteger))
                                    ? int.class
                                    : ParamClasses[i][ii-1];
                    ParamClasses[i][ii-1] =
                            (broken[i][ii].matches(regexDouble))
                                    ? double.class
                                    : ParamClasses[i][ii-1];
                    Params[i][ii-1] = broken[i][ii];
                }
            }
        }
        Object obj = null;
        Class cls = null;
        for(int i=0; i<broken.length; i++){
            try{
                if(broken[i].length==1){
                    cls = Classes.get(broken[i][0]);
                    if(Realm==null){
                        obj = cls;
                    }else{
                        obj = Realm.get(broken[i][0]);
                    }
                }else{
                    if(cls.getDeclaredMethod(broken[i][0], null) != null) {
                        Method method = cls.getDeclaredMethod(broken[i][0], ParamClasses[i]);
                        method.invoke(obj, Params[i]);
                    }
                }
            }catch(Exception ex){
                ex.printStackTrace();
            }
        }

    }

    public synchronized static Map<String, Consumer<String>> getActions(){return Actions;}

    private static Object convert(String thing){//String to objects from realms and environment...


        return null;
    }



}
