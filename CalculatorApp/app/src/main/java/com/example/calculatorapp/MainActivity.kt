package com.example.calculatorapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.calculatorapp.ui.theme.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            CalculatorAppTheme {
                CalculatorScreen()
            }
        }
    }
}

@Composable
fun CalculatorScreen() {
    var display by remember { mutableStateOf("0") }
    var operand1 by remember { mutableStateOf<Double?>(null) }
    var operator by remember { mutableStateOf<String?>(null) }
    var resetOnNextInput by remember { mutableStateOf(false) }

    fun onNumberClick(number: String) {
        if (resetOnNextInput) {
            display = number
            resetOnNextInput = false
        } else {
            display = if (display == "0" && number != ".") number
            else if (number == "." && display.contains(".")) display
            else display + number
        }
    }

    fun onOperatorClick(op: String) {
        val current = display.toDoubleOrNull() ?: return
        if (operand1 != null && operator != null && !resetOnNextInput) {
            val result = calculate(operand1!!, current, operator!!)
            display = formatResult(result)
            operand1 = result
        } else {
            operand1 = current
        }
        operator = op
        resetOnNextInput = true
    }

    fun onEqualsClick() {
        val current = display.toDoubleOrNull() ?: return
        if (operand1 != null && operator != null) {
            val result = calculate(operand1!!, current, operator!!)
            display = formatResult(result)
            operand1 = null
            operator = null
            resetOnNextInput = true
        }
    }

    fun onClearClick() {
        display = "0"
        operand1 = null
        operator = null
        resetOnNextInput = false
    }

    fun onBackspaceClick() {
        display = if (display.length > 1) display.dropLast(1) else "0"
    }

    fun onPercentClick() {
        val current = display.toDoubleOrNull() ?: return
        display = formatResult(current / 100.0)
        resetOnNextInput = true
    }

    fun onToggleSignClick() {
        val current = display.toDoubleOrNull() ?: return
        display = formatResult(current * -1)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .padding(16.dp)
            .systemBarsPadding(),
        verticalArrangement = Arrangement.Bottom
    ) {
        // Display
        Text(
            text = display,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 24.dp, end = 8.dp),
            textAlign = TextAlign.End,
            fontWeight = FontWeight.Light,
            fontSize = 64.sp,
            color = Color.White,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )

        // Button rows
        val buttonSpacing = 12.dp

        // Row 1: AC, +/-, %, ÷
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(buttonSpacing)
        ) {
            CalcButton("AC", LightGray, Color.Black, Modifier.weight(1f)) { onClearClick() }
            CalcButton("+/-", LightGray, Color.Black, Modifier.weight(1f)) { onToggleSignClick() }
            CalcButton("%", LightGray, Color.Black, Modifier.weight(1f)) { onPercentClick() }
            CalcButton("÷", Orange, Color.White, Modifier.weight(1f)) { onOperatorClick("÷") }
        }

        Spacer(modifier = Modifier.height(buttonSpacing))

        // Row 2: 7, 8, 9, ×
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(buttonSpacing)
        ) {
            CalcButton("7", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick("7") }
            CalcButton("8", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick("8") }
            CalcButton("9", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick("9") }
            CalcButton("×", Orange, Color.White, Modifier.weight(1f)) { onOperatorClick("×") }
        }

        Spacer(modifier = Modifier.height(buttonSpacing))

        // Row 3: 4, 5, 6, -
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(buttonSpacing)
        ) {
            CalcButton("4", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick("4") }
            CalcButton("5", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick("5") }
            CalcButton("6", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick("6") }
            CalcButton("-", Orange, Color.White, Modifier.weight(1f)) { onOperatorClick("-") }
        }

        Spacer(modifier = Modifier.height(buttonSpacing))

        // Row 4: 1, 2, 3, +
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(buttonSpacing)
        ) {
            CalcButton("1", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick("1") }
            CalcButton("2", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick("2") }
            CalcButton("3", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick("3") }
            CalcButton("+", Orange, Color.White, Modifier.weight(1f)) { onOperatorClick("+") }
        }

        Spacer(modifier = Modifier.height(buttonSpacing))

        // Row 5: 0 (wide), ., =
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(buttonSpacing)
        ) {
            CalcButton("0", MediumGray, Color.White, Modifier.weight(2f)) { onNumberClick("0") }
            CalcButton(".", MediumGray, Color.White, Modifier.weight(1f)) { onNumberClick(".") }
            CalcButton("=", Orange, Color.White, Modifier.weight(1f)) { onEqualsClick() }
        }
    }
}

@Composable
fun CalcButton(
    symbol: String,
    backgroundColor: Color,
    contentColor: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = modifier
            .aspectRatio(if (symbol == "0") 2.2f else 1f)
            .clip(CircleShape),
        shape = CircleShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = backgroundColor,
            contentColor = contentColor
        ),
        contentPadding = PaddingValues(0.dp)
    ) {
        Text(
            text = symbol,
            fontSize = 28.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

fun calculate(operand1: Double, operand2: Double, operator: String): Double {
    return when (operator) {
        "+" -> operand1 + operand2
        "-" -> operand1 - operand2
        "×" -> operand1 * operand2
        "÷" -> if (operand2 != 0.0) operand1 / operand2 else Double.NaN
        else -> operand2
    }
}

fun formatResult(value: Double): String {
    return if (value == value.toLong().toDouble() && !value.isNaN() && !value.isInfinite()) {
        value.toLong().toString()
    } else if (value.isNaN()) {
        "Error"
    } else {
        value.toString()
    }
}

@Preview(showBackground = true)
@Composable
fun CalculatorPreview() {
    CalculatorAppTheme {
        CalculatorScreen()
    }
}
