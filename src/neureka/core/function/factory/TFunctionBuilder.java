package neureka.core.function.factory;

import neureka.core.function.TFunction;
import neureka.core.function.imp.Constant;
import neureka.core.function.imp.Input;

import java.util.*;

public class TFunctionBuilder {
    /*
	     0:  ReLu;
		 1:  Sigmoid;
		 2:  Tanh;
		 3:  Quadratic;
		 4:  Ligmoid;
		 5:  Linear;
		 6:  Gaussian;
		 7:  abs;
		 8:  sin;
		 9:  cos;
		 10: sum;
		 11: prod;
		 12: ^;
		 13: /;
		 14: *;
		 15: %;
		 16: -;
		 17: +;
		 18: x; (conv)
	 */
    //=================================================================================================

    public static TFunction newBuild(int f_id, int size, boolean doAD){
        if (f_id == 18){
            size = 2;
        } else if(TFunction.Variables.REGISTER[f_id]==","){
            ArrayList<TFunction> srcs = new ArrayList<>();
            for(int i=0; i<size; i++){
                srcs.add(new Input().newBuild(""+i));
            }
            return TFunctionConstructor.createFunction(f_id, srcs, doAD);
        }
        if (f_id < 10) {
            return newBuild(TFunction.Variables.REGISTER[f_id] + "(I[0])", doAD);//, tipReached);
        } else if (f_id < 12) {
            return newBuild(TFunction.Variables.REGISTER[f_id] + "I[j]", doAD);//, tipReached);
        } else {
            String expression = "I[0]";
            for (int i = 0; i < size - 1; i++) {
                expression += TFunction.Variables.REGISTER[f_id] + "I[" + (i + 1) + "]";
            }
            return newBuild(expression, doAD);
        }
    }

    public static TFunction newBuild(String expression, boolean doAD) {
        HashMap<String, TFunction> map = TFunction.Variables.FUNCTIONS;
        expression =
            (expression.length()>0
                    && (expression.charAt(0) != '('||expression.charAt(expression.length() - 1) != ')'))
                        ? ("(" + expression + ")")
                        : expression;
        String k = (doAD)?"d"+expression:expression;
        if (TFunction.Variables.FUNCTIONS.containsKey(k)) {
            return TFunction.Variables.FUNCTIONS.get(k);
        }
        TFunction built = construct(expression, doAD);//, tipReached);
        if (built != null) {
            TFunction.Variables.FUNCTIONS.put(((doAD)?"d":"")+"(" + built.toString() + ")", built);
        }
        return built;
    }

    private static TFunction construct(String expression, boolean doAD){
        TFunction function = null;
        ArrayList<TFunction> sources = new ArrayList<>();;
        if (expression == null) {
            expression = "";
        }
        if (expression == "") {
            TFunction newCore = new Constant();
            newCore = newCore.newBuild("0");
            return newCore;
        }
        System.out.println("Equation packed: " + expression);
        expression = TFunctionParser.unpackAndCorrect(expression);
        System.out.println("Equation unpacked: " + expression);
        List<String> Operations = new ArrayList<String>();
        List<String> Components = new ArrayList<String>();
        int i = 0;
        System.out.println("Expression length : " + expression.length());
        while (i < expression.length()) {
            System.out.println("\nPARSING [" + expression + "] at i:" + i + "\n================");
            System.out.println("parsing at " + i + " => searching for component!");
            final String newComponent = TFunctionParser.parsedComponent(expression, i);
            if (newComponent != null) {
                System.out.println("Component found! : " + newComponent + " ");
                System.out.println(Components.size() + " < " + Operations.size() + " ?: true -> Components.e_add(" + newComponent + ")");
                if (Components.size() <= Operations.size()) {
                    Components.add(newComponent);
                }
                i += newComponent.length();
                System.out.println("parsing at " + i + " => searching for operation!");
                final String newOperation = TFunctionParser.parsedOperation(expression, i);
                if (newOperation != null) {
                    System.out.println("Operation found! : " + newOperation + " ");
                    i += newOperation.length();
                    if (newOperation.length() <= 0) {
                        continue;
                    }
                    Operations.add(newOperation);
                }
                System.out.println("parsed operation: " + newOperation);
            } else {
                ++i;
            }
            System.out.println("Ending iteration with i:" + i + "\nExpression: " + expression + "\n=============================\n----------------------------");
            Components.forEach((c) -> {
                System.out.print("[" + c + "]");
            });
            System.out.print(";\n");
            Operations.forEach((o) -> {
                System.out.print("[" + o + "]");
            });
            System.out.print(";\n");
        }
        System.out.println("Components and operations parsed: " + Components);
        System.out.println("Operations and components parsed: " + Operations);
        //===
        int Count = TFunction.Variables.REGISTER.length;
        for (int j = TFunction.Variables.REGISTER.length; j > 0; --j) {
            if (!TFunctionParser.containsOperation(TFunction.Variables.REGISTER[j - 1], Operations)) {
                --Count;
            } else {
                j = 0;
            }
        }
        int ID = 0;
        while (ID < Count) {
            final List<String> newOperations = new ArrayList<String>();
            final List<String> newComponents = new ArrayList<String>();
            System.out.println("current (" + TFunction.Variables.REGISTER[ID] + ") and most low rank operation: " + TFunction.Variables.REGISTER[Count - 1]);
            if (TFunctionParser.containsOperation(TFunction.Variables.REGISTER[ID], Operations)) {
                String currentChain = null;
                boolean groupingOccured = false;
                boolean enoughtPresent = TFunctionParser.numberOfOperationsWithin(Operations) > 1;// Otherwise: I[j]^4 goes nuts!
                if (enoughtPresent) {
                    String[] ComponentsArray = Components.toArray(new String[0]);
                    int length = ComponentsArray.length;
                    System.out.println("Iterating over " + length + " components: ");
                    for (int Ci = 0; Ci < length; Ci++) {
                        System.out.println("\nIteration " + Ci + "/" + length);
                        System.out.println("====================");
                        String currentComponent = null;
                        currentComponent = ComponentsArray[Ci];
                        String currentOperation = null;
                        if (Operations.size() > Ci) {
                            currentOperation = Operations.get(Ci);
                        }
                        System.out.println("Component: " + currentComponent);
                        System.out.println("Operation: " + currentOperation);
                        System.out.println("Chain: " + currentChain);
                        if (currentOperation != null) {
                            if (currentOperation.equals(TFunction.Variables.REGISTER[ID])) {
                                final String newChain =
                                        TFunctionParser.groupBy(TFunction.Variables.REGISTER[ID], currentChain, currentComponent, currentOperation);
                                System.out.println("newChain: " + newChain);
                                if (newChain != null) {
                                    currentChain = newChain;
                                }
                                System.out.println("This needed to be grouped");
                                groupingOccured = true;
                            } else {
                                if (currentChain == null) {
                                    newComponents.add(currentComponent);
                                } else if (currentChain != null) {
                                    newComponents.add(currentChain + currentComponent); //= String.valueOf(currentChain) + currentComponent
                                }
                                newOperations.add(currentOperation);
                                groupingOccured = true;
                                currentChain = null;
                            }
                        } else {
                            if (currentChain == null) {
                                newComponents.add(currentComponent);
                            } else {
                                newComponents.add(currentChain + currentComponent);
                                groupingOccured = true;
                            }
                            currentChain = null;
                        }
                        System.out.println("----------------\nComponent: " + currentComponent);
                        System.out.println("Operation: " + currentOperation);
                        System.out.println("Chain: " + currentChain + "\n");
                    }
                }
                if (groupingOccured) {
                    Operations = newOperations;
                    Components = newComponents;
                    System.out.println("Grouping occured:");
                } else {
                    System.out.println("Grouping did not occure:");
                }
            }
            System.out.println("ID:" + ID + " => TFunction.Variables.REGISTER[ID]: " + TFunction.Variables.REGISTER[ID]);
            System.out.println("Operations: " + Operations);
            System.out.println("Components: " + Components);
            ++ID;
        }//closed while(ID < Count)
        System.out.println("==========================================================================================================");
        //---------------------------------------------------------
        // identifying function id:
        int f_id = 0;
        if (Operations.size() >= 1) {
            System.out.println("Operation 0 : " + Operations.get(0));
            for (int k = 0; k < TFunction.Variables.REGISTER.length; ++k) {
                if (TFunction.Variables.REGISTER[k].equals(Operations.get(0))) {
                    f_id = k;
                }
            }
        }
        // building sources and function:
        if (Components.size() == 1) {
            System.out.println("Only one component left -> no operations! -> testing for function:");
            System.out.println("parsing at " + 0 + " ...(possibly a function!)");
            String possibleFunction = TFunctionParser.parsedOperation(Components.get(0).toLowerCase(), 0);
            System.out.println("TFunction ?: " + possibleFunction);
            if (possibleFunction != null) {
                if (possibleFunction.length() > 1) {
                    for (int Oi = 0; Oi < TFunction.Variables.REGISTER.length; Oi++) {
                        if (TFunction.Variables.REGISTER[Oi].equals(possibleFunction)) {
                            f_id = Oi;
                            TFunction newCore = new TFunctionBuilder()
                                .newBuild(
                                    TFunctionParser.parsedComponent(Components.get(0), possibleFunction.length()), doAD
                                );
                            sources.add(newCore);
                            function = TFunctionConstructor.createFunction(f_id, sources, doAD);
                            return function;
                            // Hallo mein Schatz! Ich bin`s, die Emi :) Ich liebe dich <3
                        }
                    }
                }
            }
            //function = TFunctionConstructor.createFunction(f_id, sources);
            System.out.println("TFunction: " + possibleFunction);
            //---
            System.out.println("1 comonent left: -> unpackAndCorrect(component)");
            String component = TFunctionParser.unpackAndCorrect(Components.get(0));
            System.out.println("component: " + component);
            System.out.println("Checking if component is variable (value/input): ");
            if ((component.charAt(0) <= '9' && component.charAt(0) >= '0') || component.charAt(0) == '-' || component.charAt(0) == '+') {
                TFunction newFunction = new Constant();
                newFunction = newFunction.newBuild(component);
                System.out.println("is value leave! -> return newValueLeave.newBuilt(component)");
                return newFunction;
            }
            if (component.charAt(0) == 'i' || component.charAt(0) == 'I' ||
                    (component.contains("[") && component.contains("]") && component.matches(".[0-9]+."))) {//TODO: Make this regex better!!
                TFunction newFunction = new Input();
                newFunction = newFunction.newBuild(component);
                System.out.println("value leave! -> return newInputLeave.newBuilt(component)");
                return newFunction;
            }
            System.out.println("Component is not f f_id Leave! -> component = TFunctionParser.cleanedHeadAndTail(component); ");
            component = TFunctionParser.cleanedHeadAndTail(component);//If the component did not trigger variable creation: =>Cleaning!
            TFunction newBuild;
            System.out.println("new build: TFunction newBuild = (TFunction)new TFunctionBuilder();");
            System.out.println("newBuild = newBuild.newBuild(component);");
            newBuild = new TFunctionBuilder().newBuild(component, doAD);
            System.out.println("-> return newBuild;");
            return newBuild;
        } else {// More than one component left:
            if (TFunction.Variables.REGISTER[f_id] == "x") {
                Components = rebindPairwise(Components, f_id);
            }else if(TFunction.Variables.REGISTER[f_id] == ","){
                if(Components.get(0).startsWith("[")){
                    Components.set(0,Components.get(0).substring(1));
                    String[] splitted;
                    if(Components.get(Components.size()-1).contains("]")){
                        int offset = 1;
                        if(Components.get(Components.size()-1).contains("]:")){
                            offset = 2;
                            splitted = Components.get(Components.size()-1).split("]:");
                        }else{
                            splitted = Components.get(Components.size()-1).split("]");
                        }

                        if(splitted.length>1){
                            splitted = new String[]{splitted[0], Components.get(Components.size()-1).substring(splitted[0].length()+offset)};
                            Components.remove(Components.size()-1);
                            for(String part : splitted){
                                Components.add(part);
                            }
                        }
                    }
                }
            }
            final ListIterator<String> ComponentIterator2 = Components.listIterator();
            while (ComponentIterator2.hasNext()) {
                final String currentComponent2 = ComponentIterator2.next();
                System.out.println("this.Input.add(newCore2.newBuild(" + currentComponent2 + ")); Input.size(): " + sources.size());
                TFunction newCore2 = new TFunctionBuilder().newBuild(currentComponent2, doAD);//Dangerous recursion lives here!
                sources.add(newCore2);
                if (newCore2 != null) {
                    System.out.println("newCore2 != null");
                }
            }
            sources.trimToSize();
            if (sources.size() == 1) {
                return sources.get(0);
            }
            if (sources.size() == 0) {
                return null;
            }
            ArrayList<TFunction> newVariable = new ArrayList<TFunction>();
            for (int Vi = 0; Vi < sources.size(); Vi++) {
                if (sources.get(Vi) != null) {
                    newVariable.add(sources.get(Vi));
                }
            }
            sources = newVariable;
            function = TFunctionConstructor.createFunction(f_id, sources, doAD);
            return function;
        }
    }

    private static List<String> rebindPairwise(List<String> components, int f_id) {
        if (components.size() > 2) {
            String newComponent = "(" + components.get(0) + TFunction.Variables.REGISTER[f_id] + components.get(1) + ")";
            components.remove(components.get(0));
            components.remove(components.get(0));
            components.add(0, newComponent);
            components = rebindPairwise(components, f_id);
        }
        return components;
    }
    //============================================================================================================================================================================================




}
