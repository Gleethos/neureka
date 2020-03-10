import neureka.Tsr
import neureka.calculus.Function

/**
 *  Operator overloading for native classes:
 */

//PLUS
Integer.metaClass.plus = ( Tsr t ) -> new Tsr(delegate.toString(),"+", t)
Double.metaClass.plus = ( Tsr t ) -> new Tsr(delegate.toString(),"+", t)
BigDecimal.metaClass.plus = ( Tsr t ) -> new Tsr(delegate.toString(),"+", t)

//MINUS
Integer.metaClass.minus = ( Tsr t ) -> new Tsr(delegate.toString(),"-", t)
Double.metaClass.minus = ( Tsr t ) -> new Tsr(delegate.toString(),"-", t)
BigDecimal.metaClass.minus = ( Tsr t ) -> new Tsr(delegate.toString(),"-", t)

//DIVIDE
Integer.metaClass.div = ( Tsr t ) -> new Tsr(delegate.toString(),"/", t)
Double.metaClass.div = ( Tsr t ) -> new Tsr(delegate.toString(),"/", t)
BigDecimal.metaClass.div = ( Tsr t ) -> new Tsr(delegate.toString(),"/", t)

//POWER
Integer.metaClass.power = ( Tsr t ) -> new Tsr(delegate.toString(),"^", t)
Double.metaClass.power = ( Tsr t ) -> new Tsr(delegate.toString(),"^", t)
BigDecimal.metaClass.power = ( Tsr t ) -> new Tsr(delegate.toString(),"^", t)

//XOR (POWER)
Integer.metaClass.xor = ( Tsr t ) -> new Tsr(delegate.toString(),"^", t)
Double.metaClass.xor = ( Tsr t ) -> new Tsr(delegate.toString(),"^", t)
BigDecimal.metaClass.xor = ( Tsr t ) -> new Tsr(delegate.toString(),"^", t)

//MULTIPLY
Integer.metaClass.multiply = ( Tsr t ) -> new Tsr(delegate.toString(),"*", t)
Double.metaClass.multiply = ( Tsr t ) -> new Tsr(delegate.toString(),"*", t)
BigDecimal.metaClass.multiply = ( Tsr t ) -> new Tsr(delegate.toString(),"*", t)

// String to Function

String.metaClass.asFunction = (boolean doAD) -> Function.create(delegate, doAD)
String.metaClass.asFunction = () -> delegate.asFunction(true)

String.metaClass.getAt = (List<Tsr> inputs)->delegate.asFunction().activate(inputs.toArray(new Tsr[0]))