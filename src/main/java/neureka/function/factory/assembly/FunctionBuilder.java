package neureka.function.factory.assembly;

import neureka.function.Function;
import neureka.function.factory.implementations.FConstant;
import neureka.function.factory.implementations.FInput;

import java.util.*;

public class FunctionBuilder {

    /**
     * @param f_id
     * @param size
     * @param doAD
     * @return
     */
    public static Function build(int f_id, int size, boolean doAD)
    {
        if (f_id == 18){
            size = 2;
        } else if(Function.TYPES.REGISTER[f_id]==","){
            ArrayList<Function> srcs = new ArrayList<>();
            for(int i=0; i<size; i++){
                srcs.add(new FInput().newBuild(""+i));
            }
            return FunctionConstructor.construct(f_id, srcs, doAD);
        }
        if (f_id < 10) {
            return build(Function.TYPES.REGISTER[f_id] + "(I[0])", doAD);
        } else if (f_id < 12) {
            return build(Function.TYPES.REGISTER[f_id] + "I[j]", doAD);
        } else {
            String expression = "I[0]";
            for (int i = 0; i < size - 1; i++) {
                expression += Function.TYPES.REGISTER[f_id] + "I[" + (i + 1) + "]";
            }
            return build(expression, doAD);
        }
    }

    /**
     * @param expression contains the function as String provided by the user
     * @param doAD is used to turn autograd on or off for this function
     * @return the function which has been built from the expression
     */
    public static Function build(String expression, boolean doAD) {
        expression =
            (expression.length()>0
                    && (expression.charAt(0) != '('||expression.charAt(expression.length() - 1) != ')'))
                        ? ("(" + expression + ")")
                        : expression;
        String k = (doAD)?"d"+expression:expression;
        if (Function.CACHE.FUNCTIONS().containsKey(k)) {
            return Function.CACHE.FUNCTIONS().get(k);
        }
        Function built = _build(expression, doAD);//, tipReached);
        if (built != null) {
            Function.CACHE.FUNCTIONS().put(((doAD)?"d":"")+"(" + built.toString() + ")", built);
        }
        return built;
    }

    /**
     * @param expression is a blueprint String for the function builder
     * @param doAD enables or disables autograd for this function
     * @return a function which has been built by the given expression
     */
    private static Function _build(String expression, boolean doAD){
        expression = expression
                        .replace("<<", ""+((char)171))
                        .replace(">>", ""+((char)187));
        expression = expression
                .replace("<-", "<")
                .replace("->", ">");
        Function function;
        ArrayList<Function> sources = new ArrayList<>();
        if (expression.equals("")) {
            Function newCore = new FConstant();
            newCore = newCore.newBuild("0");
            return newCore;
        }
        expression = FunctionParser.unpackAndCorrect(expression);
        List<String> Operations = new ArrayList<>();
        List<String> Components = new ArrayList<>();
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
        int Count = Function.TYPES.REGISTER.length;
        for (int j = Function.TYPES.REGISTER.length; j > 0; --j) {
            if (!FunctionParser.containsOperation(Function.TYPES.REGISTER[j - 1], Operations)) {
                --Count;
            } else {
                j = 0;
            }
        }
        int ID = 0;
        while (ID < Count) {
            final List<String> newOperations = new ArrayList<>();
            final List<String> newComponents = new ArrayList<>();
            if (FunctionParser.containsOperation(Function.TYPES.REGISTER[ID], Operations)) {
                String currentChain = null;
                boolean groupingOccured = false;
                boolean enoughtPresent = FunctionParser.numberOfOperationsWithin(Operations) > 1;// Otherwise: I[j]^4 goes nuts!
                if (enoughtPresent) {
                    String[] ComponentsArray = Components.toArray(new String[0]);
                    int length = ComponentsArray.length;
                    for (int Ci = 0; Ci < length; Ci++) {
                        String currentComponent;
                        currentComponent = ComponentsArray[Ci];
                        String currentOperation = null;
                        if (Operations.size() > Ci) {
                            currentOperation = Operations.get(Ci);
                        }
                        if (currentOperation != null) {
                            if (currentOperation.equals(Function.TYPES.REGISTER[ID])) {
                                final String newChain =
                                        FunctionParser.groupBy(Function.TYPES.REGISTER[ID], currentChain, currentComponent, currentOperation);
                                if (newChain != null) {
                                    currentChain = newChain;
                                }
                                groupingOccured = true;
                            } else {
                                if (currentChain == null) {
                                    newComponents.add(currentComponent);
                                } else {
                                    newComponents.add(currentChain + currentComponent); //= String.value64Of(currentChain) + currentComponent
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
            for (int k = 0; k < Function.TYPES.REGISTER.length; ++k) {
                if (Function.TYPES.REGISTER[k].equals(Operations.get(0))) {
                    f_id = k;
                }
            }
        }
        // building sources and function:
        if (Components.size() == 1) {
            String possibleFunction = FunctionParser.parsedOperation(Components.get(0).toLowerCase(), 0);
            if (possibleFunction != null) {
                if (possibleFunction.length() > 1) {
                    for (int Oi = 0; Oi < Function.TYPES.REGISTER.length; Oi++) {
                        if (Function.TYPES.REGISTER[Oi].equals(possibleFunction)) {
                            f_id = Oi;
                            Function newCore = FunctionBuilder.build(
                                    FunctionParser.parsedComponent(Components.get(0), possibleFunction.length()), doAD
                                );
                            sources.add(newCore);
                            function = FunctionConstructor.construct(f_id, sources, doAD);
                            return function;
                        }
                    }
                }
            }
            //---
            String component = FunctionParser.unpackAndCorrect(Components.get(0));
            if ((component.charAt(0) <= '9' && component.charAt(0) >= '0') || component.charAt(0) == '-' || component.charAt(0) == '+') {
                Function newFunction = new FConstant();
                newFunction = newFunction.newBuild(component);
                return newFunction;
            }
            if (component.charAt(0) == 'i' || component.charAt(0) == 'I' ||
                    (component.contains("[") && component.contains("]") && component.matches(".[0-9]+."))) {//TODO: Make this regex better!!
                Function newFunction = new FInput();
                newFunction = newFunction.newBuild(component);
                return newFunction;
            }
            String cleaned = FunctionParser.cleanedHeadAndTail(component);//If the component did not trigger variable creation: =>Cleaning!
            String raw = component.replace(cleaned, "");
            String assumed = FunctionParser.assumptionBasedOn(raw);
            if(assumed==null){
                component = cleaned;
            } else {
                component = assumed+cleaned;
            }
            Function newBuild;
            newBuild = FunctionBuilder.build(component, doAD);
            return newBuild;
        } else {// More than one component left:
            if (Function.TYPES.REGISTER[f_id].equals("x") || Function.TYPES.REGISTER[f_id].equals("<") || Function.TYPES.REGISTER[f_id].equals(">")) {
                Components = _rebindPairwise(Components, f_id);
            }else if(Function.TYPES.REGISTER[f_id].equals(",")){
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
                Function newCore2 = FunctionBuilder.build(currentComponent2, doAD);//Dangerous recursion lives here!
                sources.add(newCore2);
            }
            sources.trimToSize();
            if (sources.size() == 1) {
                return sources.get(0);
            }
            if (sources.size() == 0) {
                return null;
            }
            ArrayList<Function> newVariable = new ArrayList<Function>();
            for (int Vi = 0; Vi < sources.size(); Vi++) {
                if (sources.get(Vi) != null) {
                    newVariable.add(sources.get(Vi));
                }
            }
            sources = newVariable;
            function = FunctionConstructor.construct(f_id, sources, doAD);
            return function;
        }
    }

    /**
     * @param components
     * @param f_id
     * @return
     */
    private static List<String> _rebindPairwise(List<String> components, int f_id) {
        if (components.size() > 2) {
            String newComponent = "(" + components.get(0) + Function.TYPES.REGISTER[f_id] + components.get(1) + ")";
            components.remove(components.get(0));
            components.remove(components.get(0));
            components.add(0, newComponent);
            components = _rebindPairwise(components, f_id);
        }
        return components;
    }
    //============================================================================================================================================================================================




}
