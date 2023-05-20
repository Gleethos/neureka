import neureka.Tensor
import neureka.math.Function

/**
 *  Operator overloading for JVM native numeric types:
 */

//PLUS
Integer   .metaClass.plus = { Tensor t -> Tensor.of(delegate.toString(),"+", t ) }
Short     .metaClass.plus = { Tensor t -> Tensor.of(delegate.toString(),"+", t ) }
Byte      .metaClass.plus = { Tensor t -> Tensor.of(delegate.toString(),"+", t ) }
Double    .metaClass.plus = { Tensor t -> Tensor.of(delegate.toString(),"+", t ) }
Float     .metaClass.plus = { Tensor t -> Tensor.of(delegate.toString(),"+", t ) }
BigDecimal.metaClass.plus = { Tensor t -> Tensor.of(delegate.toString(),"+", t ) }

//MINUS
Integer   .metaClass.minus = { Tensor t -> Tensor.of(delegate.toString(),"-", t ) }
Short     .metaClass.minus = { Tensor t -> Tensor.of(delegate.toString(),"-", t ) }
Byte      .metaClass.minus = { Tensor t -> Tensor.of(delegate.toString(),"-", t ) }
Double    .metaClass.minus = { Tensor t -> Tensor.of(delegate.toString(),"-", t ) }
Float     .metaClass.minus = { Tensor t -> Tensor.of(delegate.toString(),"-", t ) }
BigDecimal.metaClass.minus = { Tensor t -> Tensor.of(delegate.toString(),"-", t ) }

//DIVIDE
Integer   .metaClass.div = { Tensor t -> Tensor.of(delegate.toString(),"/", t ) }
Short     .metaClass.div = { Tensor t -> Tensor.of(delegate.toString(),"/", t ) }
Byte      .metaClass.div = { Tensor t -> Tensor.of(delegate.toString(),"/", t ) }
Double    .metaClass.div = { Tensor t -> Tensor.of(delegate.toString(),"/", t ) }
Float     .metaClass.div = { Tensor t -> Tensor.of(delegate.toString(),"/", t ) }
BigDecimal.metaClass.div = { Tensor t -> Tensor.of(delegate.toString(),"/", t ) }

//POWER
Integer   .metaClass.power = { Tensor t -> Tensor.of(delegate.toString(),"**", t ) }
Short     .metaClass.power = { Tensor t -> Tensor.of(delegate.toString(),"**", t ) }
Byte      .metaClass.power = { Tensor t -> Tensor.of(delegate.toString(),"**", t ) }
Double    .metaClass.power = { Tensor t -> Tensor.of(delegate.toString(),"**", t ) }
Float     .metaClass.power = { Tensor t -> Tensor.of(delegate.toString(),"**", t ) }
BigDecimal.metaClass.power = { Tensor t -> Tensor.of(delegate.toString(),"**", t ) }

//XOR
Integer   .metaClass.xor = { Tensor t -> Tensor.of(delegate.toString(),"^", t ) }
Short     .metaClass.xor = { Tensor t -> Tensor.of(delegate.toString(),"^", t ) }
Byte      .metaClass.xor = { Tensor t -> Tensor.of(delegate.toString(),"^", t ) }
Double    .metaClass.xor = { Tensor t -> Tensor.of(delegate.toString(),"^", t ) }
Float     .metaClass.xor = { Tensor t -> Tensor.of(delegate.toString(),"^", t ) }
BigDecimal.metaClass.xor = { Tensor t -> Tensor.of(delegate.toString(),"^", t ) }

//MULTIPLY
Integer   .metaClass.multiply = { Tensor t -> Tensor.of(delegate.toString(),"*", t ) }
Short     .metaClass.multiply = { Tensor t -> Tensor.of(delegate.toString(),"*", t ) }
Byte      .metaClass.multiply = { Tensor t -> Tensor.of(delegate.toString(),"*", t ) }
Double    .metaClass.multiply = { Tensor t -> Tensor.of(delegate.toString(),"*", t ) }
Float     .metaClass.multiply = { Tensor t -> Tensor.of(delegate.toString(),"*", t ) }
BigDecimal.metaClass.multiply = { Tensor t -> Tensor.of(delegate.toString(),"*", t ) }

// String to Function

String.metaClass.asFunction = { boolean doAD -> Function.of(delegate, doAD) }
String.metaClass.asFunction = { delegate.asFunction( true ) }

String.metaClass.getAt = { List<Tensor> inputs -> delegate.asFunction()(inputs.toArray(new Tensor[ 0 ])) }
String.metaClass.mod = { List<Tensor> inputs -> delegate.asFunction()(inputs.toArray(new Tensor[ 0 ])) }
String.metaClass.mod = { Tensor t -> delegate.asFunction()( t ) }



