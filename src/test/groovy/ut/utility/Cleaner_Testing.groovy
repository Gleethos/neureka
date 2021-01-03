package ut.utility

import neureka.Tsr
import neureka.utility.CustomCleaner
import spock.lang.Specification

import java.lang.ref.Cleaner

class Cleaner_Testing extends Specification
{

    def 'CustomCleaner works'(){

        given :
            def cleaner = new CustomCleaner()
            def refCount = 10
            def r1 =  new Tsr([2,4])
            def r2 =  new Tsr([2,4])
            def r3 =  new Tsr([2,4])
            def r4 =  new Tsr([2,4])
            def r5 =  new Tsr([2,4])
            def r6 =  new Tsr([2,4])
            def r7 =  new Tsr([2,4])
            def r8 =  new Tsr([2,4])
            def r9 =  new Tsr([2,4])
            def r10 = new Tsr([2,4])

            cleaner.register( r1, {refCount--} )
            cleaner.register( r2, {refCount--} )
            cleaner.register( r3, {refCount--} )
            cleaner.register( r4, {refCount--} )
            cleaner.register( r5, {refCount--} )
            cleaner.register( r6, {refCount--} )
            cleaner.register( r7, {refCount--} )
            cleaner.register( r8, {refCount--} )
            cleaner.register( r9, {refCount--} )
            cleaner.register( r10,{refCount--} )

        when :
            r1 = null
            r3 = null
            System.gc()
            Thread.sleep(250)
            System.gc()
            Thread.sleep(250)

        then :
            refCount == 8
            cleaner._registered == 8

        when :
            r2 = null
            r4 = null
            System.gc()
            Thread.sleep(250)
            System.gc()
            Thread.sleep(250)

        then :
            refCount == 6
            cleaner._registered == 6

    }


}
