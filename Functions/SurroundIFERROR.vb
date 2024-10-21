' The following macro surrounds every selected sell with =IFERROR(,"")

Public Sub IFERROR0() 

Dim row As Long
Dim Col As Long
Dim FormulaString As String
Dim ReadArr As Variant

If Selection.Cells.Count > 1 Then
    ReadArr = Selection.FormulaR1C1    
    For row = LBound(ReadArr, 1) To UBound(ReadArr, 1)
        For Col = LBound(ReadArr, 2) To UBound(ReadArr, 2)        
            If Left(ReadArr(row, Col), 1) = "=" Then
            If LCase(Left(ReadArr(row, Col), 8)) <> "=iferror" Then
                ReadArr(row, Col) = "=iferror(" & Right(ReadArr(row, Col), Len(ReadArr(row, Col)) - 1) & ","""")"
            End If
            End If       
        Next
    Next   
    Selection.FormulaR1C1 = ReadArr   
    Erase ReadArr
Else
    FormulaString = Selection.FormulaR1C1
    If Left(FormulaString, 1) = "=" Then
        If LCase(Left(FormulaString, 8)) <> "=iferror" Then
            Selection.FormulaR1C1 = "=iferror(" & Right(FormulaString, Len(FormulaString) - 1) & ","""")" 
        End If
    End If
End If
End Sub