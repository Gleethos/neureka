package neureka.core.function.factory.autograd;

import neureka.core.Tsr;
import neureka.core.function.IFunction;
import neureka.core.function.factory.Function;
import neureka.core.function.factory.assembly.FunctionBuilder;

public class GraphBuilder
{
    private static IFunction MUL = FunctionBuilder.build("(I[0]*I[1])", false);
    private static IFunction ADD = FunctionBuilder.build("(I[0]+I[1])", false);

    public static void connect(Tsr output, Tsr[] inputs, IFunction function){//, boolean derive
        if(!function.isFlat()){
           return;
        }
        //--------------------------------------------------------------------------------------
        GraphLock gid = ((GraphNode)inputs[0].find(GraphNode.class)).lock();
        GraphNode node = new GraphNode(output,function, inputs, gid);
        output.add(node);
        if(node.usesAD() && function.isFlat()){
            /**  Preparing for back propagation:  * */
            if(node.usesForwardAD()){
                int i = 0;
                for(Tsr input : inputs){
                    GraphNode src_node = ((GraphNode) input.find(GraphNode.class));
                    if(src_node.function()!=null && src_node.function().id()==IFunction.TYPES.LOOKUP.get("x")){
                        Tsr d = function.derive(inputs, i);//TODO: is this ever used? / visited? - yes but why?
                        node.put(input, d);// Sources created by x-mul are revers-mode cases!
                    }else{
                        if(src_node.usesAD()){
                            Tsr d = function.derive(inputs, i);
                            if(src_node.size()==0 && node.size()==0){
                                node.put(inputs[i], d);
                            } else {
                                src_node.forEach(
                                    (t, g)->{
                                    /**
                                     *  Chain rule (forward) for every
                                     *  _gradient w.r.t. leaves (reverseAD or user leaves):
                                     * */
                                    if(node.has(t)){
                                        Tsr dg = node.get(t);
                                        node.put(t, ADD.activate(new Tsr[]{dg, MUL.activate(new Tsr[]{d, g})}));
                                    }else{
                                        node.put(t, MUL.activate(new Tsr[]{d, g}));
                                    }//TODO: flag within src tsrs that grant that the tensor has been created by function constructor!
                                });
                            }
                        }
                    }
                    i++;
                }
            }else if(node.usesReverseAD()) {
                int i = 0;
                for(Tsr input : inputs){
                    GraphNode src_node = ((GraphNode) input.find(GraphNode.class));
                    if(src_node.mode()!=0 || input.rqsGradient()){
                        Tsr d = function.derive(inputs, i);
                        node.put(input, d);// Add gradients with respect to every source tensor!
                    }
                    i++;
                }
            }
        }
    }
    //--------------------------------------------------------------------------------------

}



