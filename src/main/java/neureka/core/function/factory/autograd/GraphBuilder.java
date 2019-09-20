package neureka.core.function.factory.autograd;

import neureka.core.T;
import neureka.core.function.IFunction;
import neureka.core.function.factory.Function;

public class GraphBuilder
{
    public static void connect(T drain, T[] source, IFunction function){//, boolean derive
        if(!function.isFlat()){
           return;
        }
        //--------------------------------------------------------------------------------------
        GraphLock gid = ((GraphNode)source[0].find(GraphNode.class)).lock();
        GraphNode node = new GraphNode(drain,function, source, gid);
        drain.add(node);
        if(node.usesAD() && function.isFlat()){
            /**  Preparing for back propagation:  * */
            if(node.usesForwardAD()){
                int i = 0;
                for(T src : source){
                    GraphNode src_node = ((GraphNode) src.find(GraphNode.class));
                    if(src_node.function()!=null && src_node.function().id()==18){
                        T d = function.derive(source, i);
                        node.put(src, d);// Sources created by x-mul are revers-mode cases!
                    }else{
                        if(src_node.usesAD()){
                            T d = function.derive(source, i);
                            if(src_node.size()==0 && node.size()==0){
                                node.put(source[i], d);
                            } else {
                                src_node.forEach(
                                    (t, g)->{
                                    /**
                                     *  Chain rule (forward) for every
                                     *  _gradient w.r.t. leaves (reverseAD or user leaves):
                                     * */
                                    if(node.has(t)){
                                        T dg = node.get(t);
                                        node.put(t, Function.exec.addition(dg, Function.exec.multiplication(d, g)));
                                    }else{
                                        node.put(t, Function.exec.multiplication(d, g));
                                    }
                                    //TODO: flag within src tsrs that grant that the tensor has been created by function constructor!
                                });
                            }

                        }
                        i++;
                    }
                }
            }else if(node.usesReverseAD()) {
                int i = 0;
                for(T src : source){
                    GraphNode src_node = ((GraphNode) src.find(GraphNode.class));
                    if(src_node.mode()!=0 || src.rqsGradient()){
                        T d = function.derive(source, i);
                        node.put(src, d);// Add gradients with respect to every source tensor!
                    }
                    i++;
                }
            }
        }
    }
    //--------------------------------------------------------------------------------------

}



