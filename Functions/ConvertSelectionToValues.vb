' The following macro will convert the selection to its values. In other words this is the equivalent to running =VALUE(CELL) on the selection.
' NOTE: THIS WILL REPLACE ALL CELLS AND YOU CAN'T UNDO

Sub ConvertSelectionToValues()
    Dim c As Range
    Dim response As VbMsgBoxResult
    
    ' Check if a range is selected
    If TypeName(Selection) <> "Range" Then
        MsgBox "Please select a range of cells first.", vbExclamation
        Exit Sub
    End If
    
    ' Warn the user and ask for confirmation
    response = MsgBox("Warning: This action will replace the values in the selected cells and cannot be undone." & vbNewLine & _
                      "Do you want to continue?", vbYesNo + vbExclamation, "Confirm Action")
    
    ' If the user clicks No, exit the sub
    If response = vbNo Then
        Exit Sub
    End If
    
    Application.ScreenUpdating = False
    
    For Each c In Selection
        If Not IsEmpty(c) Then
            On Error Resume Next
            ' Convert the cell's content to a number
            c.Value = c.Value
            On Error GoTo 0
        End If
    Next c
    
    Application.ScreenUpdating = True
    
    MsgBox "Conversion completed!", vbInformation
End Sub