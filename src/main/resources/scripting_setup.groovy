
import neureka.Tsr
import neureka.calculus.Function

/**
 *  Operator overloading for native classes:
 */

//PLUS
Integer.metaClass.plus = { Tsr t -> Tsr.of(delegate.toString(),"+", t ) }
Double.metaClass.plus = { Tsr t -> Tsr.of(delegate.toString(),"+", t ) }
BigDecimal.metaClass.plus = { Tsr t -> Tsr.of(delegate.toString(),"+", t ) }

//MINUS
Integer.metaClass.minus = { Tsr t -> Tsr.of(delegate.toString(),"-", t ) }
Double.metaClass.minus = { Tsr t -> Tsr.of(delegate.toString(),"-", t ) }
BigDecimal.metaClass.minus = { Tsr t -> Tsr.of(delegate.toString(),"-", t ) }

//DIVIDE
Integer.metaClass.div = { Tsr t -> Tsr.of(delegate.toString(),"/", t ) }
Double.metaClass.div = { Tsr t -> Tsr.of(delegate.toString(),"/", t ) }
BigDecimal.metaClass.div = { Tsr t -> Tsr.of(delegate.toString(),"/", t ) }

//POWER
Integer.metaClass.power = { Tsr t -> Tsr.of(delegate.toString(),"^", t ) }
Double.metaClass.power = { Tsr t -> Tsr.of(delegate.toString(),"^", t ) }
BigDecimal.metaClass.power = { Tsr t -> Tsr.of(delegate.toString(),"^", t ) }

//XOR (POWER)
Integer.metaClass.xor = { Tsr t -> Tsr.of(delegate.toString(),"^", t ) }
Double.metaClass.xor = { Tsr t -> Tsr.of(delegate.toString(),"^", t ) }
BigDecimal.metaClass.xor = { Tsr t -> Tsr.of(delegate.toString(),"^", t ) }

//MULTIPLY
Integer.metaClass.multiply = { Tsr t -> Tsr.of(delegate.toString(),"*", t ) }
Double.metaClass.multiply = { Tsr t -> Tsr.of(delegate.toString(),"*", t ) }
BigDecimal.metaClass.multiply = { Tsr t -> Tsr.of(delegate.toString(),"*", t ) }

// String to Function

String.metaClass.asFunction = { boolean doAD -> Function.create(delegate, doAD) }
String.metaClass.asFunction = { delegate.asFunction( true ) }

String.metaClass.getAt = { List<Tsr> inputs -> delegate.asFunction()(inputs.toArray(new Tsr[ 0 ])) }
String.metaClass.mod = { List<Tsr> inputs -> delegate.asFunction()(inputs.toArray(new Tsr[ 0 ])) }
String.metaClass.mod = { Tsr t -> delegate.asFunction()( t ) }



