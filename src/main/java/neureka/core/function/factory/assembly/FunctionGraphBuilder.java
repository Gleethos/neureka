package neureka.core.function.factory.assembly;

import neureka.core.function.IFunction;
import neureka.core.function.factory.implementations.FConstant;
import neureka.core.function.factory.implementations.FInput;

import java.util.*;

public class FunctionGraphBuilder {

    /**
     * @param f_id
     * @param size
     * @param doAD
     * @return
     */
    public static IFunction newBuild(int f_id, int size, boolean doAD){
        if (f_id == 18){
            size = 2;
        } else if(IFunction.TYPES.REGISTER[f_id]==","){
            ArrayList<IFunction> srcs = new ArrayList<>();
            for(int i=0; i<size; i++){
                srcs.add(new FInput().newBuild(""+i));
            }
            return FunctionConstructor.createFunction(f_id, srcs, doAD);
        }
        if (f_id < 10) {
            return newBuild(IFunction.TYPES.REGISTER[f_id] + "(I[0])", doAD);//, tipReached);
        } else if (f_id < 12) {
            return newBuild(IFunction.TYPES.REGISTER[f_id] + "I[j]", doAD);//, tipReached);
        } else {
            String expression = "I[0]";
            for (int i = 0; i < size - 1; i++) {
                expression += IFunction.TYPES.REGISTER[f_id] + "I[" + (i + 1) + "]";
            }
            return newBuild(expression, doAD);
        }
    }

    /**
     * @param expression
     * @param doAD
     * @return
     */
    public static IFunction newBuild(String expression, boolean doAD) {
        expression =
            (expression.length()>0
                    && (expression.charAt(0) != '('||expression.charAt(expression.length() - 1) != ')'))
                        ? ("(" + expression + ")")
                        : expression;
        String k = (doAD)?"d"+expression:expression;
        if (IFunction.CACHE.FUNCTIONS().containsKey(k)) {
            return IFunction.CACHE.FUNCTIONS().get(k);
        }
        IFunction built = construct(expression, doAD);//, tipReached);
        if (built != null) {
            IFunction.CACHE.FUNCTIONS().put(((doAD)?"d":"")+"(" + built.toString() + ")", built);
        }
        return built;
    }

    /**
     * @param expression
     * @param doAD
     * @return
     */
    private static IFunction construct(String expression, boolean doAD){
        expression = expression
                        .replace("<<", ""+((char)171))
                        .replace(">>", ""+((char)187));
        expression = expression
                .replace("<-", "<")
                .replace("->", ">");
        IFunction function = null;
        ArrayList<IFunction> sources = new ArrayList<>();;
        if (expression == null) {
            expression = "";
        }
        if (expression == "") {
            IFunction newCore = new FConstant();
            newCore = newCore.newBuild("0");
            return newCore;
        }
        expression = FunctionParser.unpackAndCorrect(expression);
        List<String> Operations = new ArrayList<String>();
        List<String> Components = new ArrayList<String>();
        int i = 0;
        while (i < expression.length()) {
            final String newComponent = FunctionParser.parsedComponent(expression, i);
            if (newComponent != null) {
                if (Components.size() <= Operations.size()) {
                    Components.add(newComponent);
                }
                i += newComponent.length();
                final String newOperation = FunctionParser.parsedOperation(expression, i);
                if (newOperation != null) {
                    i += newOperation.length();
                    if (newOperation.length() <= 0) {
                        continue;
                    }
                    Operations.add(newOperation);
                }
            } else {
                ++i;
            }
        }
        //===
        int Count = IFunction.TYPES.REGISTER.length;
        for (int j = IFunction.TYPES.REGISTER.length; j > 0; --j) {
            if (!FunctionParser.containsOperation(IFunction.TYPES.REGISTER[j - 1], Operations)) {
                --Count;
            } else {
                j = 0;
            }
        }
        int ID = 0;
        while (ID < Count) {
            final List<String> newOperations = new ArrayList<String>();
            final List<String> newComponents = new ArrayList<String>();
            if (FunctionParser.containsOperation(IFunction.TYPES.REGISTER[ID], Operations)) {
                String currentChain = null;
                boolean groupingOccured = false;
                boolean enoughtPresent = FunctionParser.numberOfOperationsWithin(Operations) > 1;// Otherwise: I[j]^4 goes nuts!
                if (enoughtPresent) {
                    String[] ComponentsArray = Components.toArray(new String[0]);
                    int length = ComponentsArray.length;
                    for (int Ci = 0; Ci < length; Ci++) {
                        String currentComponent = null;
                        currentComponent = ComponentsArray[Ci];
                        String currentOperation = null;
                        if (Operations.size() > Ci) {
                            currentOperation = Operations.get(Ci);
                        }
                        if (currentOperation != null) {
                            if (currentOperation.equals(IFunction.TYPES.REGISTER[ID])) {
                                final String newChain =
                                        FunctionParser.groupBy(IFunction.TYPES.REGISTER[ID], currentChain, currentComponent, currentOperation);
                                if (newChain != null) {
                                    currentChain = newChain;
                                }
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
                    }
                }
                if (groupingOccured) {
                    Operations = newOperations;
                    Components = newComponents;
                }
            }
            ++ID;
        }//closed while(ID < Count)
        //---------------------------------------------------------
        // identifying function id:
        int f_id = 0;
        if (Operations.size() >= 1) {
            for (int k = 0; k < IFunction.TYPES.REGISTER.length; ++k) {
                if (IFunction.TYPES.REGISTER[k].equals(Operations.get(0))) {
                    f_id = k;
                }
            }
        }
        // building sources and function:
        if (Components.size() == 1) {
            String possibleFunction = FunctionParser.parsedOperation(Components.get(0).toLowerCase(), 0);
            if (possibleFunction != null) {
                if (possibleFunction.length() > 1) {
                    for (int Oi = 0; Oi < IFunction.TYPES.REGISTER.length; Oi++) {
                        if (IFunction.TYPES.REGISTER[Oi].equals(possibleFunction)) {
                            f_id = Oi;
                            IFunction newCore = FunctionGraphBuilder.newBuild(
                                    FunctionParser.parsedComponent(Components.get(0), possibleFunction.length()), doAD
                                );
                            sources.add(newCore);
                            function = FunctionConstructor.createFunction(f_id, sources, doAD);
                            return function;
                            // I <3 u 2
                        }
                    }
                }
            }
            //---
            String component = FunctionParser.unpackAndCorrect(Components.get(0));
            if ((component.charAt(0) <= '9' && component.charAt(0) >= '0') || component.charAt(0) == '-' || component.charAt(0) == '+') {
                IFunction newFunction = new FConstant();
                newFunction = newFunction.newBuild(component);
                return newFunction;
            }
            if (component.charAt(0) == 'i' || component.charAt(0) == 'I' ||
                    (component.contains("[") && component.contains("]") && component.matches(".[0-9]+."))) {//TODO: Make this regex better!!
                IFunction newFunction = new FInput();
                newFunction = newFunction.newBuild(component);
                return newFunction;
            }
            component = FunctionParser.cleanedHeadAndTail(component);//If the component did not trigger variable creation: =>Cleaning!
            IFunction newBuild;
            newBuild = FunctionGraphBuilder.newBuild(component, doAD);
            return newBuild;
        } else {// More than one component left:
            if (IFunction.TYPES.REGISTER[f_id] == "x" || IFunction.TYPES.REGISTER[f_id]=="<" || IFunction.TYPES.REGISTER[f_id]==">") {
                Components = rebindPairwise(Components, f_id);
            }else if(IFunction.TYPES.REGISTER[f_id] == ","){
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
                IFunction newCore2 = FunctionGraphBuilder.newBuild(currentComponent2, doAD);//Dangerous recursion lives here!
                sources.add(newCore2);
            }
            sources.trimToSize();
            if (sources.size() == 1) {
                return sources.get(0);
            }
            if (sources.size() == 0) {
                return null;
            }
            ArrayList<IFunction> newVariable = new ArrayList<IFunction>();
            for (int Vi = 0; Vi < sources.size(); Vi++) {
                if (sources.get(Vi) != null) {
                    newVariable.add(sources.get(Vi));
                }
            }
            sources = newVariable;
            function = FunctionConstructor.createFunction(f_id, sources, doAD);
            return function;
        }
    }

    /**
     * @param components
     * @param f_id
     * @return
     */
    private static List<String> rebindPairwise(List<String> components, int f_id) {
        if (components.size() > 2) {
            String newComponent = "(" + components.get(0) + IFunction.TYPES.REGISTER[f_id] + components.get(1) + ")";
            components.remove(components.get(0));
            components.remove(components.get(0));
            components.add(0, newComponent);
            components = rebindPairwise(components, f_id);
        }
        return components;
    }
    //============================================================================================================================================================================================




}
