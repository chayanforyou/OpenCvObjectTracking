package io.github.chayanforyou.tracking

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) :
    View(context, attrs) {

    private val callbacks = mutableListOf<DrawCallback>()

    fun addCallback(callback: DrawCallback) {
        callbacks.add(callback)
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        synchronized(this) {
            callbacks.forEach { it.drawCallback(canvas) }
        }
    }

    interface DrawCallback {
        fun drawCallback(canvas: Canvas)
    }
}